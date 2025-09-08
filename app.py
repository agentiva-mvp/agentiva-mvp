# app.py
import os, json, re, unicodedata
import numpy as np
import streamlit as st
from typing import List, Dict, Any
from urllib.parse import quote, quote_plus
from datetime import datetime

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# =======================
# Utilities & Config
# =======================
def env(name, default=None):
    return st.secrets.get(name, os.getenv(name, default))

def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return s

YES_TOKENS = {
    "ja","j","yes","y","ok","okay","okey","gern","gerne","passt","passt so","klingt gut",
    "mach","mach mal","ist ok","ist okay","alles klar","go","bitte","ja bitte","sounds good","go for it"
}
NO_TOKENS = {"nein","n","no","nope","lass","lass sein","abbrechen","spater","später","nicht","ne","nee"}
FALLBACK_HINT_TOKENS = {
    "hilfe","hilf","unterstutze","unterstütze","anfrage","mail schreiben","mail verfassen",
    "formuliere","formuliere eine anfrage","kontakt aufnehmen","ansprechpartner","zustandiger","zuständiger"
}

def is_yes(text: str) -> bool: return any(tok in _norm(text) for tok in YES_TOKENS)
def is_no(text: str)  -> bool: return any(tok in _norm(text) for tok in NO_TOKENS)
def wants_fallback(text: str) -> bool: return any(tok in _norm(text) for tok in FALLBACK_HINT_TOKENS)

def build_mailto(to: str, subject: str, body: str) -> str:
    from urllib.parse import quote
    def enc(x: str) -> str: return quote(x or "")
    q = []
    if subject: q.append(f"subject={enc(subject)}")
    if body:    q.append(f"body={enc(body)}")
    qs = "&".join(q)
    return f"mailto:{enc(to or '')}" + (f"?{qs}" if qs else "")

# Deeplinks via PDF.js
GITHUB_OWNER  = env("GITHUB_OWNER", "")
GITHUB_REPO   = env("GITHUB_REPO", "")
GITHUB_BRANCH = env("GITHUB_BRANCH", "main")

def pdf_raw_url(filename: str, page: int | None = None) -> str:
    if not (GITHUB_OWNER and GITHUB_REPO):
        local = f"./unterlagen/{quote(filename, safe='')}"
        return f"{local}#page={page}" if page else local
    enc_file = quote(filename, safe="")
    raw = f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/{GITHUB_BRANCH}/unterlagen/{enc_file}"
    enc_raw = quote_plus(raw)
    url = f"https://mozilla.github.io/pdf.js/web/viewer.html?file={enc_raw}"
    return f"{url}#page={page}" if page else url

# Azure
AZURE_OPENAI_KEY = env("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = env("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_CHAT_DEPLOYMENT = env("AZURE_OPENAI_CHAT_DEPLOYMENT")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = env("AZURE_OPENAI_API_VERSION", "2024-06-01")

if not AZURE_OPENAI_KEY or not AZURE_OPENAI_ENDPOINT:
    st.stop()

# Paths
INDEX_DIR   = "agentiva_db"
INDEX_NPZ   = os.path.join(INDEX_DIR, "index.npz")
INDEX_META  = os.path.join(INDEX_DIR, "metadaten.json")
INDEX_CONT  = os.path.join(INDEX_DIR, "contacts.json")
INDEX_INFO  = os.path.join(INDEX_DIR, "index_info.json")
DOCS_DIR    = "unterlagen"

# Thresholds
FALLBACK_MIN_SIM  = float(env("FALLBACK_MIN_SIM", "0.35"))
FALLBACK_MIN_GOOD = int(env("FALLBACK_MIN_GOOD", "1"))

# =======================
# Streamlit UI header
# =======================
st.set_page_config(page_title="Agentiva – Wissenslotse", page_icon="🧭", layout="wide")
st.title("🤖 Agentiva – Ihr Wissenslotse")
st.caption("Antwortet faktenbasiert aus PDFs. Bei Unklarheit startet Agentiva einen dialogbasierten Fallback und formuliert eine E-Mail-Anfrage. Quellen (Snippet + Deeplink) nur bei starken Treffern.")

# =======================
# Azure Clients
# =======================
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

# =======================
# Index Build & Load
# =======================
def extract_emails_with_snippets(text: str, window: int = 120) -> List[Dict[str, str]]:
    out = []
    for m in re.finditer(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text):
        start = max(0, m.start() - window)
        end   = min(len(text), m.end() + window)
        snippet = " ".join(text[start:end].split())
        out.append({"email": m.group(0).lower(), "snippet": snippet})
    return out

def file_manifest(root: str) -> dict:
    out = {}
    if not os.path.isdir(root): return out
    for name in os.listdir(root):
        if name.lower().endswith(".pdf"):
            p = os.path.join(root, name)
            try:
                st_ = os.stat(p)
                out[name] = {"size": st_.st_size, "mtime": st_.st_mtime}
            except Exception:
                pass
    return out

def save_index_info(manifest: dict):
    os.makedirs(INDEX_DIR, exist_ok=True)
    with open(INDEX_INFO, "w", encoding="utf-8") as f:
        json.dump({"built_at": datetime.now().isoformat(), "files": manifest}, f)

def load_index_info() -> dict:
    if not os.path.exists(INDEX_INFO): return {}
    try:
        with open(INDEX_INFO, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

REQUIRED_META_KEYS = {"source", "chunk_id", "text", "page"}

def load_index() -> tuple[np.ndarray | None, list | None, list, dict, str]:
    reason = ""
    contacts = []
    info = load_index_info()

    if not (os.path.exists(INDEX_NPZ) and os.path.exists(INDEX_META)):
        return None, None, contacts, info, "missing_files"

    # Check manifest drift
    cur = file_manifest(DOCS_DIR)
    if (info or {}).get("files", {}) != (cur or {}):
        return None, None, contacts, {}, "stale_manifest"

    try:
        data = np.load(INDEX_NPZ)
        E = data["embeddings"].astype(np.float32)
    except Exception:
        return None, None, contacts, info, "npz_load_error"

    try:
        with open(INDEX_META, "r", encoding="utf-8") as f:
            M = json.load(f)
    except Exception:
        return None, None, contacts, info, "meta_load_error"

    # contacts optional
    if os.path.exists(INDEX_CONT):
        try:
            with open(INDEX_CONT, "r", encoding="utf-8") as f:
                contacts = json.load(f)
        except Exception:
            contacts = []

    if not isinstance(M, list) or len(M) == 0:
        return None, None, contacts, info, "meta_empty"

    if E.shape[0] != len(M):
        return None, None, contacts, info, "length_mismatch"

    # required keys?
    sample_n = min(20, len(M))
    for i in range(sample_n):
        if not REQUIRED_META_KEYS.issubset(set((M[i] or {}).keys())):
            return None, None, contacts, info, "missing_keys"

    return E, M, contacts, info, ""

def build_index_now() -> tuple[np.ndarray | None, list | None, list]:
    from pypdf import PdfReader

    def chunk_text(text: str, size: int = 950, overlap: int = 150):
        text = " ".join(text.split())
        chunks, start, n = [], 0, len(text)
        while start < n:
            end = min(start + size, n)
            chunks.append(text[start:end])
            if end == n: break
            start = max(0, end - overlap)
        return chunks

    pdfs = [os.path.join(DOCS_DIR, p) for p in os.listdir(DOCS_DIR) if p.lower().endswith(".pdf")] if os.path.isdir(DOCS_DIR) else []
    if not pdfs:
        st.error(f"Keine PDFs in '{DOCS_DIR}/' gefunden.")
        return None, None, []

    all_chunks, meta = [], []
    email_map: Dict[str, Dict[str, Any]] = {}

    for path in pdfs:
        base = os.path.basename(path)
        try:
            reader = PdfReader(path)
        except Exception:
            continue

        for page_idx, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            if not text.strip(): continue

            # index chunks
            for i, ch in enumerate(chunk_text(text)):
                all_chunks.append(ch)
                meta.append({"source": base, "chunk_id": int(f"{page_idx:05d}{i:03d}"), "text": ch, "page": page_idx})

            # contacts
            for rec in extract_emails_with_snippets(text):
                e = rec["email"]
                if e not in email_map:
                    email_map[e] = {"email": e, "snippets": []}
                email_map[e]["snippets"].append({"source": base, "page": page_idx, "snippet": rec["snippet"]})

    if not all_chunks:
        return None, None, []

    # Embeddings
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
    with open(INDEX_CONT, "w", encoding="utf-8") as f:
        json.dump(list(email_map.values()), f, ensure_ascii=False, indent=2)

    save_index_info(file_manifest(DOCS_DIR))
    return E, meta, list(email_map.values())

E, META, CONTACTS, INFO, reason = load_index()
if E is None or META is None:
    with st.spinner("Baue Wissensindex…"):
        E, META, CONTACTS = build_index_now()

# =======================
# Sidebar
# =======================
with st.sidebar:
    st.subheader("⚙️ Index & Diagnose")
    if E is not None and META is not None:
        st.success(f"Aktiver Index: {len(META)} Chunks")
    else:
        st.warning("Kein Index geladen.")

    built_at = (INFO or {}).get("built_at")
    if built_at:
        try:
            dt = datetime.fromisoformat(built_at).strftime("%Y-%m-%d %H:%M")
            st.caption(f"📅 Zuletzt gebaut: {dt}")
        except Exception:
            st.caption(f"📅 Zuletzt gebaut: {built_at}")

    if st.button("🔄 Index jetzt neu bauen"):
        with st.spinner("Baue Wissensindex…"):
            E, META, CONTACTS = build_index_now()
            if E is not None: st.success(f"Index gebaut: {len(META)} Chunks")
            else: st.error("Index konnte nicht gebaut werden.")

    st.divider()
    DEBUG = st.checkbox("🔍 Debug (Scores & Relevanzcheck)", value=False)

    if st.button("🗑️ Chat zurücksetzen"):
        st.session_state.clear()
        st.rerun()

# =======================
# Retrieval & Helpers
# =======================
def embed_query_vec(qtext: str) -> np.ndarray:
    v = np.array(emb.embed_query(qtext), dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-10)

def pdf_hits(query: str, top_k: int = 4) -> List[Dict[str, Any]]:
    if E is None or META is None: return []
    q = embed_query_vec(query)
    sims = (E @ q)
    idx = np.argsort(-sims)[:top_k]
    hits = []
    for i in idx:
        it = dict(META[i])
        it["score"] = float(sims[i])
        hits.append(it)
    return hits

def pick_best_contact(question: str, thema: str, kontext: str) -> Dict[str, Any] | None:
    if not CONTACTS:
        return None
    query_text = " ".join([question or "", thema or "", kontext or ""]).strip() or (question or thema or kontext or "")
    q = embed_query_vec(query_text)
    best, best_score = None, -1.0
    for c in CONTACTS:
        snippets = c.get("snippets", [])[:3]
        if not snippets: continue
        joined = " ".join(s.get("snippet","") for s in snippets)
        dv = np.array(emb.embed_documents([joined])[0], dtype=np.float32)
        dv /= (np.linalg.norm(dv) + 1e-10)
        score = float(dv @ q)
        if score > best_score:
            best_score, best = score, {"email": c.get("email"), "score": score, "snippets": snippets}
    return best

def llm_relevance_check(question: str, hits: List[Dict[str, Any]]) -> bool:
    """Semantischer Check: passen die Kontexte wirklich zur Frage?"""
    try:
        contexts = "\n\n".join([(h.get("text","") or "")[:300] for h in hits])
        prompt = (
            "Beurteile nüchtern, ob die folgenden Kontexte fachlich zur Frage passen.\n"
            "Antworte nur mit 'JA' oder 'NEIN'.\n\n"
            f"Frage:\n{question}\n\nKontexte:\n{contexts}"
        )
        resp = llm.invoke([{"role":"user","content":prompt}]).content.strip().lower()
        return "ja" in resp
    except Exception:
        # Bei Fehler lieber konservativ True → keine unnötigen Fallbacks
        return True

def should_fallback(question: str, hits: List[Dict[str, Any]]) -> bool:
    if not hits: return True
    good = [h for h in hits if float(h.get("score",0)) >= FALLBACK_MIN_SIM]
    # zu wenige gute Treffer?
    if len(good) < FALLBACK_MIN_GOOD:
        return True
    # semantisch unpassend?
    return not llm_relevance_check(question, good)

def answer_with_context(question: str, passages: List[Dict[str, Any]], history: List[Dict[str,str]]) -> str:
    # kleine Conversation-Memory (letzte 6 Nachrichten)
    hist_txt = ""
    for turn in history[-6:]:
        role = "Nutzer" if turn["role"] == "user" else "Agentiva"
        hist_txt += f"{role}: {turn['content']}\n"
    blocks = "\n\n".join([f"[{p.get('source','?')} • Seite {p.get('page','?')}] {(p.get('text','') or '')[:300]}" for p in passages])
    system = (
        "Du bist Agentiva, ein Wissenslotse. Antworte präzise, knapp und ausschließlich faktenbasiert aus den Kontexte. "
        "Wenn die Kontexte nicht ausreichen, sage das ehrlich."
    )
    user_msg = f"Verlauf (Auszug):\n{hist_txt}\n\nFrage:\n{question}\n\nKontexte:\n{blocks}"
    msg = llm.invoke([{"role":"system","content":system},{"role":"user","content":user_msg}])
    return msg.content

def make_email_draft(question: str, details: dict) -> dict:
    subject = details.get("thema") or f"Klärungsanfrage: {question[:60]}"
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

# =======================
# Session State
# =======================
if "messages" not in st.session_state: st.session_state.messages = []
if "fallback_step" not in st.session_state: st.session_state.fallback_step = None
if "fallback_data" not in st.session_state: st.session_state.fallback_data = {}
if "await_snippet_confirm" not in st.session_state: st.session_state.await_snippet_confirm = False
if "last_hits" not in st.session_state: st.session_state.last_hits = []
if "deferred_user_input" not in st.session_state: st.session_state.deferred_user_input = None
if "suggested_contact" not in st.session_state: st.session_state.suggested_contact = None

# =======================
# Chat UI – render history
# =======================
st.subheader("💬 Chat")
for m in st.session_state.messages:
    role = "user" if m["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.markdown(m["content"])

user_input = st.chat_input("Schreibe deine Frage…")

# Vorrang für Deferred Input (z. B. aus beendetem Quellen-Dialog)
if st.session_state.deferred_user_input and not user_input:
    user_input = st.session_state.deferred_user_input
    st.session_state.deferred_user_input = None

# =======================
# Snippet-Handler
# =======================
if user_input and st.session_state.await_snippet_confirm:
    st.session_state.messages.append({"role":"user","content":user_input})
    if is_yes(user_input):
        lines = []
        for h in st.session_state.last_hits:
            url = pdf_raw_url(h.get('source',''), h.get('page'))
            snippet = (h.get("text","") or "")[:200].replace("\n"," ")
            lines.append(f"- **{h.get('source','?')} – Seite {h.get('page','?')}** → [Original öffnen]({url})\n\n> {snippet}…")
        st.session_state.messages.append({"role":"assistant","content":"Hier sind die passenden Stellen:\n" + "\n".join(lines)})
        st.session_state.await_snippet_confirm = False
        st.rerun()
    elif is_no(user_input):
        st.session_state.messages.append({"role":"assistant","content":"Alles klar, keine Quellenanzeige."})
        st.session_state.await_snippet_confirm = False
        st.rerun()
    else:
        # neue Frage → Quellenmodus beenden und Frage normal verarbeiten
        st.session_state.await_snippet_confirm = False
        st.session_state.deferred_user_input = user_input
        st.rerun()

# =======================
# Fallback-Dialog
# =======================
def handle_fallback_dialog(user_input: str):
    step = st.session_state.fallback_step
    data = st.session_state.fallback_data

    if step == "ask_consent":
        if is_yes(user_input):
            st.session_state.fallback_step = "ask_thema"
            return {"role":"assistant","content":"Alles klar. Worum geht es genau (Thema)?"}
        elif is_no(user_input):
            st.session_state.fallback_step = None
            return {"role":"assistant","content":"Verstanden, dann belassen wir es dabei."}
        else:
            return {"role":"assistant","content":"Sag bitte **ja** zum Start oder **nein** zum Abbrechen."}

    elif step == "ask_thema":
        data["thema"] = user_input
        st.session_state.fallback_step = "ask_ziel"
        return {"role":"assistant","content":"Danke. Was soll mit der Anfrage erreicht werden (Ziel/Ergebnis)?"}

    elif step == "ask_ziel":
        data["ziel"] = user_input
        st.session_state.fallback_step = "ask_frist"
        return {"role":"assistant","content":"Gibt es eine Frist oder einen Termin?"}

    elif step == "ask_frist":
        data["frist"] = user_input
        st.session_state.fallback_step = "ask_kontext"
        return {"role":"assistant","content":"Gibt es Kontext/Hintergrund, den ich erwähnen soll?"}

    elif step == "ask_kontext":
        data["kontext"] = user_input
        # Ansprechpartner-Vorschlag
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
                    "content": f"Ich schlage **{best.get('email','?')}** als Ansprechpartner vor.\n"
                               f"Bezug aus den Unterlagen:\n{details or '-'}\n\n"
                               "Passt das? Antworte mit **ja**, **nein** oder gib eine andere E-Mail an."}
        else:
            return {"role":"assistant",
                    "content":"Ich habe keinen Ansprechpartner erkannt. "
                             "Bitte gib eine E-Mail-Adresse an oder antworte **nein**."}

    elif step == "confirm_contact":
        txt = user_input.strip()
        best = st.session_state.suggested_contact
        if is_yes(txt) and best:
            data["recipient"] = best.get("email","kontakt@example.com")
        elif "@" in _norm(txt):
            data["recipient"] = txt
        elif is_no(txt):
            st.session_state.fallback_step = None
            st.session_state.suggested_contact = None
            return {"role":"assistant","content":"Okay, kein Versand. Du kannst später erneut fragen."}
        else:
            return {"role":"assistant","content":"Bitte **ja**, **nein** oder eine E-Mail-Adresse angeben."}

        st.session_state.fallback_step = "done"
        draft = make_email_draft(data.get("original_question",""), data)
        st.session_state.fallback_data["draft"] = draft
        return {"role":"assistant",
                "content": f"Fertig! Soll ich die E-Mail an **{data['recipient']}** vorbereiten? "
                           "Antworte mit **ja** oder **nein**."}

    elif step == "done":
        if is_yes(user_input):
            draft = data.get("draft", {"subject":"", "body":""})
            to_addr = data.get("recipient","kontakt@example.com")
            mailto_url = build_mailto(to=to_addr, subject=draft.get("subject",""), body=draft.get("body",""))
            st.session_state.fallback_step = None
            st.session_state.suggested_contact = None
            return {"role":"assistant","content": f"[📧 Im Mailprogramm öffnen]({mailto_url})"}
        elif is_no(user_input):
            st.session_state.fallback_step = None
            st.session_state.suggested_contact = None
            return {"role":"assistant","content":"Alles klar, kein Versand."}
        else:
            return {"role":"assistant","content":"Bitte **ja** oder **nein**."}

# =======================
# Haupt-Chatlogik
# =======================
if user_input and not st.session_state.await_snippet_confirm:
    st.session_state.messages.append({"role":"user","content":user_input})

    # Falls wir mitten im Fallback-Dialog sind → zuerst bedienen
    if st.session_state.fallback_step:
        reply = handle_fallback_dialog(user_input)
        st.session_state.messages.append(reply)
        st.rerun()

    # Expliziter Fallbackwunsch (unabhängig von Scores)
    if wants_fallback(user_input):
        st.session_state.fallback_step = "ask_thema"
        st.session_state.fallback_data = {"original_question": user_input}
        st.session_state.messages.append({"role":"assistant","content":"Alles klar. Worum geht es genau (Thema)?"})
        st.rerun()

    # Reguläre Antwort mit Retrieval
    if E is None or META is None:
        st.session_state.messages.append({"role":"assistant","content":"Kein Index gefunden. Baue den Index in der Sidebar neu."})
        st.rerun()

    hits = pdf_hits(user_input, top_k=4)

    # Debug
    if DEBUG:
        debug_lines = "\n".join([f"- {h.get('source','?')} S.{h.get('page','?')} | Score: {float(h.get('score',0)):.3f}" for h in hits])
        st.session_state.messages.append({"role":"assistant","content":"**Debug – Treffer & Scores:**\n" + debug_lines})

    # Fallback-Entscheidung (Score + Semantik)
    if should_fallback(user_input, hits):
        st.session_state.fallback_step = "ask_consent"
        st.session_state.fallback_data = {"original_question": user_input}
        st.session_state.messages.append({"role":"assistant","content":"Ich finde keine eindeutige Antwort. Soll ich dir helfen, eine Anfrage zu formulieren?"})
        st.rerun()

    # Antwort mit starken, relevanten Treffern
    strong = [h for h in hits if float(h.get("score",0)) >= FALLBACK_MIN_SIM]
    answer = answer_with_context(user_input, strong, st.session_state.messages)
    st.session_state.messages.append({"role":"assistant","content":answer})

    # Quellen optional
    st.session_state.messages.append({"role":"assistant","content":"Möchtest du die entsprechenden Stellen im Original sehen? Antworte mit **ja**."})
    st.session_state.await_snippet_confirm = True
    st.session_state.last_hits = strong
    st.rerun()
