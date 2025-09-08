# app.py
import os, json, re
import numpy as np
import streamlit as st
from typing import List, Dict, Any
from urllib.parse import quote, quote_plus
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# ---------- Helpers ----------
def env(name, default=None):
    return st.secrets.get(name, os.getenv(name, default))

def build_mailto(to: str, subject: str, body: str, cc: str = "", bcc: str = "") -> str:
    """Baut einen mailto:-Link, der den lokalen Mail-Client öffnet."""
    def enc(x: str) -> str: return quote(x or "")
    to_enc = enc(to or "")
    q = []
    if subject: q.append(f"subject={enc(subject)}")
    if body:    q.append(f"body={enc(body)}")
    if cc:      q.append(f"cc={enc(cc)}")
    if bcc:     q.append(f"bcc={enc(bcc)}")
    qs = "&".join(q)
    return f"mailto:{to_enc}" + (f"?{qs}" if qs else "")

def extract_emails_with_snippets(text: str, window: int = 120) -> List[Dict[str, str]]:
    """Finde E-Mails und liefere kleine Text-Umfelder als Snippets."""
    out = []
    for m in re.finditer(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text):
        start = max(0, m.start() - window)
        end   = min(len(text), m.end() + window)
        snippet = " ".join(text[start:end].split())
        out.append({"email": m.group(0).lower(), "snippet": snippet})
    return out

# Deeplink zu GitHub-PDF (Mozilla PDF.js)
GITHUB_OWNER  = env("GITHUB_OWNER", "")
GITHUB_REPO   = env("GITHUB_REPO", "")
GITHUB_BRANCH = env("GITHUB_BRANCH", "main")

def pdf_raw_url(filename: str, page: int | None = None) -> str:
    """Deeplink via Mozilla PDF.js auf eine konkrete Seite."""
    if not (GITHUB_OWNER and GITHUB_REPO):
        local = f"./unterlagen/{quote(filename, safe='')}"
        return f"{local}#page={page}" if page else local
    enc_file = quote(filename, safe="")
    raw = f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/{GITHUB_BRANCH}/unterlagen/{enc_file}"
    enc_raw = quote_plus(raw)
    url = f"https://mozilla.github.io/pdf.js/web/viewer.html?file={enc_raw}"
    return f"{url}#page={page}" if page else url

# ---------- Azure / App Config ----------
AZURE_OPENAI_KEY = env("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = env("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_CHAT_DEPLOYMENT = env("AZURE_OPENAI_CHAT_DEPLOYMENT")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = env("AZURE_OPENAI_API_VERSION", "2024-06-01")

if not AZURE_OPENAI_KEY or not AZURE_OPENAI_ENDPOINT:
    st.stop()

INDEX_DIR  = "agentiva_db"
INDEX_NPZ  = os.path.join(INDEX_DIR, "index.npz")
INDEX_META = os.path.join(INDEX_DIR, "metadaten.json")
INDEX_CONTACTS = os.path.join(INDEX_DIR, "contacts.json")
UNTERLAGEN_DIR = "unterlagen"

st.set_page_config(page_title="Agentiva – Wissenslotse", page_icon="🧭", layout="wide")
st.title("🤖 Agentiva – Ihr Wissenslotse")
st.caption("Antwortet faktenbasiert aus PDFs. Fallback formuliert im Chat eine Anfrage und öffnet dein Mailprogramm vorbefüllt. Quellen (Mini-Snippets + Deeplink) nur bei starken Treffern.")

# ---------- Azure Clients ----------
llm = AzureChatOpenAI(
    azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    temperature=0.2,
)
emb = AzureOpenAIEmbeddings(
    azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
)

# ---------- Index laden/aufbauen ----------
def load_index():
    if os.path.exists(INDEX_NPZ) and os.path.exists(INDEX_META):
        data = np.load(INDEX_NPZ)
        E = data["embeddings"].astype(np.float32)
        with open(INDEX_META, "r", encoding="utf-8") as f:
            M = json.load(f)
        contacts = []
        if os.path.exists(INDEX_CONTACTS):
            with open(INDEX_CONTACTS, "r", encoding="utf-8") as f:
                contacts = json.load(f)
        return E, M, contacts
    return None, None, []

def build_index_now():
    from pypdf import PdfReader

    def chunk_text(text: str, size: int = 900, overlap: int = 150):
        text = " ".join(text.split())
        chunks, start, n = [], 0, len(text)
        while start < n:
            end = min(start + size, n)
            chunks.append(text[start:end])
            if end == n: break
            start = max(0, end - overlap)
        return chunks

    pdfs = [os.path.join(UNTERLAGEN_DIR, p) for p in os.listdir(UNTERLAGEN_DIR) if p.lower().endswith(".pdf")]
    if not pdfs:
        st.error(f"Keine PDFs in '{UNTERLAGEN_DIR}/' gefunden.")
        return None, None, []

    all_chunks, meta = [], []
    email_map: Dict[str, Dict[str, Any]] = {}

    for p in pdfs:
        basename = os.path.basename(p)
        try:
            reader = PdfReader(p)
        except Exception:
            continue

        for page_idx, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
            if not page_text.strip():
                continue

            for i, ch in enumerate(chunk_text(page_text)):
                all_chunks.append(ch)
                meta.append({
                    "source": basename,
                    "chunk_id": int(f"{page_idx:05d}{i:03d}"),
                    "text": ch,
                    "page": page_idx
                })

            for rec in extract_emails_with_snippets(page_text):
                e = rec["email"]
                if e not in email_map:
                    email_map[e] = {"email": e, "snippets": []}
                email_map[e]["snippets"].append({
                    "source": basename,
                    "page": page_idx,
                    "snippet": rec["snippet"]
                })

    if not all_chunks:
        return None, None, []
    vecs, BATCH = [], 64
    for i in range(0, len(all_chunks), BATCH):
        batch = all_chunks[i:i+BATCH]
        vecs.extend(emb.embed_documents(batch))
    E = np.array(vecs, dtype=np.float32)
    E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-10)

    os.makedirs(INDEX_DIR, exist_ok=True)
    np.savez_compressed(INDEX_NPZ, embeddings=E)
    with open(INDEX_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    contacts = list(email_map.values())
    with open(INDEX_CONTACTS, "w", encoding="utf-8") as f:
        json.dump(contacts, f, ensure_ascii=False, indent=2)
    return E, meta, contacts

E, META, CONTACTS = load_index()
# ---------- Sidebar ----------
with st.sidebar:
    st.subheader("⚙️ Index & Diagnose")
    if E is not None:
        st.success(f"Aktiver Index: {len(META)} Chunks")
    else:
        st.warning("Kein Index geladen.")

    if st.button("🔄 Index jetzt neu bauen"):
        with st.spinner("Baue Wissensindex…"):
            E, META, CONTACTS = build_index_now()
            if E is not None: st.success(f"Index gebaut: {len(META)} Chunks")
            else: st.error("Index konnte nicht gebaut werden.")

    st.divider()
    DEBUG = st.checkbox("🔍 Debug-Modus (Scores anzeigen)", value=False)

    if st.button("🗑️ Chat zurücksetzen"):
        st.session_state.clear()
        st.rerun()

# ---------- Retrieval ----------
def pdf_hits(query: str, top_k: int = 4) -> List[Dict[str, Any]]:
    if E is None or META is None: return []
    q = np.array(emb.embed_query(query), dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-10)
    sims = (E @ q)
    idx = np.argsort(-sims)[:top_k]
    hits = []
    for i in idx:
        item = dict(META[i])
        item["score"] = float(sims[i])
        hits.append(item)
    return hits

# ---------- Fallback-Entscheidung ----------
FALLBACK_MIN_SIM = float(env("FALLBACK_MIN_SIM", "0.35"))
FALLBACK_MIN_GOOD = int(env("FALLBACK_MIN_GOOD", "1"))

def should_fallback(hits: List[Dict[str, Any]]) -> bool:
    if not hits: return True
    good = sum(1 for h in hits if h["score"] >= FALLBACK_MIN_SIM)
    return good < FALLBACK_MIN_GOOD

# ---------- Auto-Contact-Ranking ----------
def pick_best_contact(question: str, thema: str, kontext: str) -> Dict[str, Any] | None:
    if not CONTACTS:
        return None
    query_text = " ".join([question or "", thema or "", kontext or ""]).strip()
    if not query_text:
        query_text = question or thema or kontext or ""
    q_emb = np.array(emb.embed_query(query_text), dtype=np.float32)
    q_emb /= (np.linalg.norm(q_emb) + 1e-10)

    best = None
    best_score = -1.0
    for c in CONTACTS:
        snippets = c.get("snippets", [])[:3]
        if not snippets: continue
        joined = " ".join(s.get("snippet","") for s in snippets)
        doc_emb = np.array(emb.embed_documents([joined])[0], dtype=np.float32)
        doc_emb /= (np.linalg.norm(doc_emb) + 1e-10)
        score = float(doc_emb @ q_emb)
        if score > best_score:
            best_score = score
            best = {"email": c["email"], "score": score, "snippets": snippets}
    return best

# ---------- Antwortgenerator ----------
def answer_with_context(question: str, passages: List[Dict[str, Any]]) -> str:
    blocks = []
    for p in passages:
        blocks.append(f"[{p['source']} • Seite {p['page']}] {p['text'][:300]}")
    context = "\n\n".join(blocks)

    system = (
        "Du bist Agentiva, ein Wissenslotse. Antworte faktenbasiert aus den bereitgestellten Kontexten, "
        "knapp, klar und in deutscher Sprache."
    )
    user_msg = f"Frage:\n{question}\n\nKontexte:\n{context}"
    msg = llm.invoke([{"role":"system","content":system},{"role":"user","content":user_msg}])
    return msg.content

def make_email_draft(question: str, details: dict) -> dict:
    subject = details.get("thema") or f"Klärungsanfrage: {question[:40]}"
    body = (
        f"Hallo,\n\n"
        f"ich habe eine Frage zu: {details.get('thema','(Thema)')}\n\n"
        f"Hintergrund:\n{details.get('kontext','')}\n\n"
        f"Meine konkrete Frage:\n{question}\n\n"
        f"Erwartetes Ergebnis/Ziel:\n{details.get('ziel','')}\n"
        f"Frist/Zeitpunkt: {details.get('frist','')}\n\n"
        f"Vielen Dank!\n"
    )
    return {"subject": subject, "body": body}

# ---------- Session State ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "fallback_step" not in st.session_state:
    st.session_state.fallback_step = None
if "fallback_data" not in st.session_state:
    st.session_state.fallback_data = {}
if "await_snippet_confirm" not in st.session_state:
    st.session_state.await_snippet_confirm = False
if "last_hits" not in st.session_state:
    st.session_state.last_hits = []
if "suggested_contact" not in st.session_state:
    st.session_state.suggested_contact = None

# ---------- Chat UI ----------
st.subheader("💬 Chat")

for msg in st.session_state.messages:
    role = "Du" if msg["role"] == "user" else "Agentiva"
    icon = "👤" if msg["role"] == "user" else "🤖"
    st.markdown(f"{icon} **{role}:** {msg['content']}")

user_input = st.chat_input("Schreibe deine Frage…")

# ---------- Snippet-Handler (nur bei starken Treffern) ----------
if user_input and st.session_state.await_snippet_confirm:
    st.session_state.await_snippet_confirm = False
    st.session_state.messages.append({"role":"user","content":user_input})
    yes_words = {"ja","yes","gern","j","ok","okay","zeige","zeigen","bitte","please"}

    if user_input.strip().lower() in yes_words:
        lines = []
        for h in st.session_state.last_hits:
            url = pdf_raw_url(h['source'], h['page'])
            snippet = h["text"][:200].replace("\n"," ")
            lines.append(f"- **{h['source']} – Seite {h['page']}** → [Original öffnen]({url})\n\n> {snippet}…")
        st.session_state.messages.append({
            "role":"assistant",
            "content": "Hier sind die passenden Stellen im Original:\n" + "\n".join(lines)
        })
    else:
        st.session_state.messages.append({"role":"assistant","content":"Alles klar, keine Quellenanzeige."})
    st.rerun()

# ---------- Hauptlogik ----------
def handle_fallback_dialog(user_input: str):
    step = st.session_state.fallback_step
    data = st.session_state.fallback_data

    if step == "ask_consent":
        if user_input.lower() in {"ja","yes","ok","okay","gern"}:
            st.session_state.fallback_step = "ask_thema"
            return {"role":"assistant","content":"Alles klar. Worum geht es genau (Thema)?"}
        else:
            st.session_state.fallback_step = None
            return {"role":"assistant","content":"Verstanden, dann belassen wir es dabei."}

    elif step == "ask_thema":
        data["thema"] = user_input
        st.session_state.fallback_step = "ask_ziel"
        return {"role":"assistant","content":"Danke. Was soll mit der Anfrage erreicht werden (Ziel/Ergebnis)?"}

    elif step == "ask_ziel":
        data["ziel"] = user_input
        st.session_state.fallback_step = "ask_frist"
        return {"role":"assistant","content":"Gut. Gibt es eine Frist oder einen Termin?"}

    elif step == "ask_frist":
        data["frist"] = user_input
        st.session_state.fallback_step = "ask_kontext"
        return {"role":"assistant","content":"Noch kurz: gibt es einen Kontext/Hintergrund, den ich erwähnen soll?"}

    elif step == "ask_kontext":
        data["kontext"] = user_input
        # automatischer Ansprechpartner-Vorschlag
        best = pick_best_contact(
            question=data.get("original_question",""),
            thema=data.get("thema",""),
            kontext=data.get("kontext","")
        )
        st.session_state.suggested_contact = best
        st.session_state.fallback_step = "confirm_contact"

        if best:
            details = ""
            for s in best.get("snippets", [])[:2]:
                details += f"- {s.get('source','')} (S. {s.get('page','?')})\n"
            return {"role":"assistant",
                    "content": f"Ich schlage **{best['email']}** als Ansprechpartner vor.\n"
                               f"Bezug aus den Unterlagen:\n{details or '-'}\n\n"
                               "Passt das? Antworte mit **ja**, **nein** oder gib eine andere E-Mail an."}
        else:
            return {"role":"assistant",
                    "content":"Ich habe keinen Ansprechpartner in den Unterlagen erkannt. "
                             "Bitte gib eine E-Mail-Adresse an oder schreibe **nein** zum Abbrechen."}

    elif step == "confirm_contact":
        txt = user_input.strip()
        best = st.session_state.suggested_contact
        if txt.lower() in {"ja","yes","ok","okay","passt"} and best:
            data["recipient"] = best["email"]
        elif "@" in txt:
            data["recipient"] = txt
        else:
            st.session_state.fallback_step = None
            st.session_state.suggested_contact = None
            return {"role":"assistant","content":"Okay, kein Versand. Du kannst später erneut fragen."}

        st.session_state.fallback_step = "done"
        draft = make_email_draft(data.get("original_question",""), data)
        st.session_state.fallback_data["draft"] = draft
        return {"role":"assistant",
                "content": f"Fertig! Soll ich die E-Mail an **{data['recipient']}** vorbereiten? "
                           "Antworte mit **ja** oder **nein**."}

    elif step == "done":
        if user_input.lower() in {"ja","yes","ok","okay"}:
            draft = data.get("draft")
            to_addr = data.get("recipient","kontakt@example.com")
            mailto_url = build_mailto(to=to_addr, subject=draft["subject"], body=draft["body"])
            st.session_state.fallback_step = None
            st.session_state.suggested_contact = None
            return {"role":"assistant","content": f"[📧 Im Mailprogramm öffnen]({mailto_url})"}
        else:
            st.session_state.fallback_step = None
            st.session_state.suggested_contact = None
            return {"role":"assistant","content":"Alles klar, kein Versand."}

if user_input and not st.session_state.await_snippet_confirm:
    st.session_state.messages.append({"role":"user","content":user_input})

    # Wenn wir mitten im Fallback-Dialog sind, zuerst den bedienen
    if st.session_state.fallback_step:
        reply = handle_fallback_dialog(user_input)
        st.session_state.messages.append(reply)
        st.rerun()

    # Sonst regulär versuchen zu beantworten
    if E is None:
        st.session_state.messages.append({"role":"assistant","content":"Kein Index gefunden. Baue den Index in der Sidebar neu."})
        st.rerun()

    hits = pdf_hits(user_input, top_k=4)

    # Debug: Scores zeigen
    if DEBUG:
        debug_lines = "\n".join([f"- {h['source']} S.{h['page']} | Score: {h['score']:.3f}" for h in hits])
        st.session_state.messages.append({"role":"assistant","content":"**Debug – Treffer & Scores:**\n" + debug_lines})

    # Fallback?
    if should_fallback(hits):
        st.session_state.fallback_step = "ask_consent"
        st.session_state.fallback_data = {"original_question": user_input}
        st.session_state.messages.append({
            "role":"assistant",
            "content":"Ich finde keine eindeutige Antwort in den Dokumenten. "
                      "Soll ich dir helfen, eine Anfrage zu formulieren?"
        })
        st.rerun()

    # Starke Treffer → antworten + Quellen optional
    strong = [h for h in hits if h["score"] >= FALLBACK_MIN_SIM]
    if not strong:
        # Sicherheitsnetz: wenn strong leer, trotzdem Fallback
        st.session_state.fallback_step = "ask_consent"
        st.session_state.fallback_data = {"original_question": user_input}
        st.session_state.messages.append({
            "role":"assistant",
            "content":"Ich finde keine eindeutige Antwort in den Dokumenten. "
                      "Soll ich dir helfen, eine Anfrage zu formulieren?"
        })
        st.rerun()

    blocks = "\n\n".join([f"[{h['source']} • Seite {h['page']}] {h['text'][:300]}" for h in strong])
    system = "Du bist ein Assistent, der faktenbasiert aus den Dokumenten antwortet."
    user_msg = f"Frage:\n{user_input}\n\nKontexte:\n{blocks}"
    msg = llm.invoke([{"role":"system","content":system},{"role":"user","content":user_msg}])
    st.session_state.messages.append({"role":"assistant","content":msg.content})

    # Nachfrage Quellen
    st.session_state.messages.append({
        "role":"assistant",
        "content":"Möchtest du die entsprechenden Stellen im Original sehen? Antworte mit **ja**.",
    })
    st.session_state.await_snippet_confirm = True
    st.session_state.last_hits = strong
    st.rerun()
