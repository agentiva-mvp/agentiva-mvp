# app.py
import os, json, re
import numpy as np
import streamlit as st
from datetime import datetime
from typing import List, Dict, Any
from urllib.parse import quote, quote_plus
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# ---------- Helpers ----------
def env(name, default=None):
    return st.secrets.get(name, os.getenv(name, default))

def build_mailto(to: str, subject: str, body: str, cc: str = "", bcc: str = "") -> str:
    """Baut einen mailto:-Link, der den lokalen Mail-Client des Nutzers öffnet."""
    def enc(x: str) -> str:
        return quote(x or "")
    to_enc = enc(to or "")
    q = []
    if subject: q.append(f"subject={enc(subject)}")
    if body:    q.append(f"body={enc(body)}")
    if cc:      q.append(f"cc={enc(cc)}")
    if bcc:     q.append(f"bcc={enc(bcc)}")
    qs = "&".join(q)
    return f"mailto:{to_enc}" + (f"?{qs}" if qs else "")

# ---------- Azure / App Config ----------
AZURE_OPENAI_KEY = env("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = env("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_CHAT_DEPLOYMENT = env("AZURE_OPENAI_CHAT_DEPLOYMENT")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = env("AZURE_OPENAI_API_VERSION", "2024-06-01")

# GitHub Repo für PDF-Links (PDF.js lädt die RAW-URL)
GITHUB_OWNER  = env("GITHUB_OWNER", "")
GITHUB_REPO   = env("GITHUB_REPO", "")
GITHUB_BRANCH = env("GITHUB_BRANCH", "main")

if not AZURE_OPENAI_KEY or not AZURE_OPENAI_ENDPOINT:
    st.stop()

INDEX_DIR  = "agentiva_db"
INDEX_NPZ  = os.path.join(INDEX_DIR, "index.npz")
INDEX_META = os.path.join(INDEX_DIR, "metadaten.json")
INDEX_INFO = os.path.join(INDEX_DIR, "index_info.json")
UNTERLAGEN_DIR = "unterlagen"

st.set_page_config(page_title="Agentiva – Wissenslotse", page_icon="🧭", layout="wide")
st.title("🤖 Agentiva – Ihr Wissenslotse")
st.caption("Antwortet faktenbasiert aus PDFs. Zeigt auf Wunsch Textausschnitte inkl. Deeplink ins Original (PDF.js). Fallback erstellt strukturierte E-Mails und öffnet dein Mailprogramm vorbefüllt.")

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

# ---------- Manifest & Index-Status ----------
def file_manifest(root: str) -> dict:
    manifest = {}
    if not os.path.isdir(root):
        return manifest
    for name in os.listdir(root):
        if not name.lower().endswith(".pdf"):
            continue
        p = os.path.join(root, name)
        try:
            st_ = os.stat(p)
            manifest[name] = {"size": st_.st_size, "mtime": st_.st_mtime}
        except Exception:
            continue
    return manifest

def is_index_stale(info: dict, current: dict) -> bool:
    return (info or {}).get("files", {}) != current

# ---------- Deeplink-Builder (Mozilla PDF.js) ----------
def pdf_raw_url(filename: str, page: int | None = None) -> str:
    """
    Deeplink via Mozilla PDF.js:
      https://mozilla.github.io/pdf.js/web/viewer.html?file=<RAW_URL>#page=N
    Voraussetzungen: Öffentliches GitHub-Repo.
    """
    owner, repo, branch = GITHUB_OWNER, GITHUB_REPO, GITHUB_BRANCH
    if not owner or not repo:
        local = f"./unterlagen/{quote(filename, safe='')}"
        return f"{local}#page={page}" if page else local
    enc_file = quote(filename, safe="")
    raw = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/unterlagen/{enc_file}"
    enc_raw = quote_plus(raw)
    url = f"https://mozilla.github.io/pdf.js/web/viewer.html?file={enc_raw}"
    return f"{url}#page={page}" if page else url

# ---------- Index laden/aufbauen ----------
def load_index():
    if os.path.exists(INDEX_NPZ) and os.path.exists(INDEX_META):
        data = np.load(INDEX_NPZ)
        E = data["embeddings"].astype(np.float32)
        with open(INDEX_META, "r", encoding="utf-8") as f:
            M = json.load(f)
        info = {}
        if os.path.exists(INDEX_INFO):
            try:
                with open(INDEX_INFO, "r", encoding="utf-8") as f:
                    info = json.load(f)
            except Exception:
                info = {}
        # Backfill
        for m in M:
            m.setdefault("last_modified", None)
            m.setdefault("page", None)
            m.setdefault("text", "")
        return E, M, info
    return None, None, {}

def build_index_now():
    from pypdf import PdfReader

    def chunk_text(text: str, size: int = 900, overlap: int = 150):
        text = " ".join(text.split())
        chunks, start, n = [], 0, len(text)
        while start < n:
            end = min(start + size, n)
            chunks.append(text[start:end])
            if end == n:
                break
            start = max(0, end - overlap)
        return chunks

    pdfs = [os.path.join(UNTERLAGEN_DIR, p) for p in os.listdir(UNTERLAGEN_DIR) if p.lower().endswith(".pdf")]
    if not pdfs:
        st.error(f"Keine PDFs in '{UNTERLAGEN_DIR}/' gefunden.")
        return None, None

    all_chunks: List[str] = []
    meta: List[Dict[str, Any]] = []

    for p in pdfs:
        basename = os.path.basename(p)
        try:
            reader = PdfReader(p)
        except Exception:
            continue

        try:
            mtime = os.path.getmtime(p)
        except Exception:
            mtime = None

        # seitenweises Chunking -> saubere Seitenzahl für Deeplinks
        for page_idx, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
            if not page_text.strip():
                continue
            chunks = chunk_text(page_text, size=900, overlap=150)
            for i, ch in enumerate(chunks):
                all_chunks.append(ch)
                meta.append({
                    "source": basename,
                    "chunk_id": int(f"{page_idx:05d}{i:03d}"),
                    "text": ch,
                    "last_modified": mtime,
                    "page": page_idx
                })

    # Embeddings
    vecs, BATCH = [], 64
    for i in range(0, len(all_chunks), BATCH):
        batch = all_chunks[i:i+BATCH]
        vecs.extend(emb.embed_documents(batch))
    if not vecs:
        return None, None

    E = np.array(vecs, dtype=np.float32)
    E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-10)

    os.makedirs(INDEX_DIR, exist_ok=True)
    np.savez_compressed(INDEX_NPZ, embeddings=E)
    with open(INDEX_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    current = file_manifest(UNTERLAGEN_DIR)
    with open(INDEX_INFO, "w", encoding="utf-8") as f:
        json.dump({
            "built_at": datetime.now().isoformat(),
            "files": current
        }, f)

    return E, meta

# ---------- Auto-(Re)Index ----------
E, META, INFO = load_index()
current = file_manifest(UNTERLAGEN_DIR)
if (E is None) or is_index_stale(INFO, current):
    with st.spinner("Änderungen in 'unterlagen/' erkannt – baue Index neu…"):
        E, META = build_index_now()
        _, _, INFO = load_index()

# ---------- Kontakte aus PDFs extrahieren ----------
def extract_contacts(meta: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    contacts = {}
    email_re = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
    for m in meta or []:
        text = m.get("text", "") or ""
        for match in email_re.finditer(text):
            email = match.group(0)
            start = max(0, match.start() - 80)
            snippet = text[start:match.end() + 40].replace("\n", " ").strip()
            label = snippet if len(snippet) <= 120 else snippet[:120] + "…"
            contacts[email] = {"email": email, "label": label, "source": m.get("source",""), "page": m.get("page")}
    return list(contacts.values())

CONTACTS = extract_contacts(META)

# ---------- Sidebar ----------
with st.sidebar:
    st.subheader("⚙️ Index")
    if E is not None:
        st.info(f"Aktiver Index: {E.shape[0]} Chunks")
    built_at = INFO.get("built_at")
    if built_at:
        try:
            dt = datetime.fromisoformat(built_at).strftime("%Y-%m-%d %H:%M")
            st.caption(f"📅 Index zuletzt aktualisiert: {dt}")
        except Exception:
            st.caption(f"📅 Index zuletzt aktualisiert: {built_at}")
    st.divider()
    if st.button("🗑️ Chat zurücksetzen"):
        st.session_state.clear()
        st.rerun()

# ---------- Retrieval & Answering ----------
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
        page = item.get("page")
        item["url"] = pdf_raw_url(item["source"], page)
        hits.append(item)
    return hits

def answer_with_context(question: str, passages: List[Dict[str, Any]]) -> str:
    blocks = []
    for p in passages:
        preview = p.get("text", "")[:900]
        page = p.get("page")
        blocks.append(f"[{p['source']} • Seite {page}] {preview}")
    context = "\n\n".join(blocks)

    system = (
        "Du bist Agentiva, ein Wissenslotse: ein KI-Assistent, der Anwendern hilft, Antworten in komplexen Dokumenten zu finden. "
        "Antworte stets faktenbasiert und klar auf Deutsch. "
        "Dein Ziel ist es, dem Nutzer so gut wie möglich eine finale, hilfreiche Antwort zu geben. "
        "Wenn die Antwort nicht eindeutig aus den bereitgestellten Kontexten hervorgeht, prüfe, ob im Kontext Ansprechpartner, "
        "Kontaktdaten oder Zuständigkeiten genannt sind. Falls ja, schlage diese als Ansprechpartner vor. "
        "Wenn keine Ansprechpartner genannt sind, sage ehrlich, dass im Dokument keine passenden Ansprechpartner gefunden wurden."
    )
    user_msg = f"Frage:\n{question}\n\nKontexte:\n{context}"
    msg = llm.invoke([{"role":"system","content":system},{"role":"user","content":user_msg}])
    return msg.content

def should_fallback(hits: List[Dict[str, Any]], min_sim: float = 0.25) -> bool:
    if not hits: return True
    best = max(h["score"] for h in hits)
    return best < min_sim

def make_email_draft(question: str, details: dict) -> dict:
    subject = details.get("betreff") or f"Klärungsanfrage: {details.get('thema','Anliegen')}"
    anrede  = details.get("anrede") or "Hallo,"
    ziel    = details.get("ziel") or "Bitte um Rückmeldung"
    frist   = details.get("frist") or "—"
    kontext = details.get("kontext") or ""
    body = (
        f"{anrede}\n\n"
        f"ich habe eine Frage zu: {details.get('thema','(Thema)')}\n\n"
        f"Hintergrund:\n{kontext}\n\n"
        f"Meine konkrete Frage:\n{question}\n\n"
        f"Erwartetes Ergebnis/Ziel:\n{ziel}\n"
        f"Frist/Zeitpunkt: {frist}\n\n"
        f"Vielen Dank!\n"
    )
    return {"subject": subject, "body": body}

# ---------- Chat State ----------
if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, Any]] = []
if "await_snippet_confirm" not in st.session_state:
    st.session_state.await_snippet_confirm = False
if "last_hits" not in st.session_state:
    st.session_state.last_hits: List[Dict[str, Any]] = []
if "fallback_mode" not in st.session_state:
    st.session_state.fallback_mode = False
if "fallback_data" not in st.session_state:
    st.session_state.fallback_data = {}

st.subheader("💬 Chat")

# Verlauf rendern
for msg in st.session_state.messages:
    role = "Du" if msg["role"] == "user" else "Agentiva"
    icon = "👤" if msg["role"] == "user" else "🤖"
    st.markdown(f"{icon} **{role}:** {msg['content']}")
    if msg.get("snippets"):
        with st.expander("📚 Quellen & Ausschnitte"):
            for h in msg["snippets"]:
                when = ""
                if h.get("last_modified"):
                    try:
                        doc_dt = datetime.fromtimestamp(h["last_modified"]).strftime("%Y-%m-%d")
                        when = f" • geändert: {doc_dt}"
                    except Exception:
                        pass
                page = h.get("page")
                st.markdown(
                    f"**{h['source']} – Seite {page}** (Score {h['score']:.3f}{when})\n\n"
                    f"> {h.get('text','')[:600]}..."
                )
                if h.get("url"):
                    st.markdown(f"[📄 Original öffnen (Seite {page})]({h['url']})")

# Eingabe
user_input = st.chat_input("Schreibe deine Frage… (oder antworte mit „ja“ um Quellen zu sehen)")
if user_input:
    # A) Nutzer antwortet auf „Quellen anzeigen?“
    if st.session_state.await_snippet_confirm:
        st.session_state.await_snippet_confirm = False
        yes_words = {"ja","yes","gern","j","ok","okay","zeige","zeigen","bitte","please","yo"}
        st.session_state.messages.append({"role": "user", "content": user_input})
        if user_input.strip().lower() in yes_words:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Hier sind die relevanten Textausschnitte mit Link ins Original:",
                "snippets": st.session_state.last_hits
            })
        else:
            st.session_state.messages.append({"role": "assistant", "content": "Alles klar. Wenn du später die Quelle sehen möchtest, sag einfach „zeige Quelle“."})
        st.rerun()

    # B) normaler Durchlauf (Frage)
    st.session_state.messages.append({"role": "user", "content": user_input})
    if E is None:
        st.session_state.messages.append({"role": "assistant", "content": "Kein Index gefunden. Lade bitte Dokumente hoch oder baue den Index neu."})
        st.rerun()

    with st.spinner("Suche relevante Passagen…"):
        hits = pdf_hits(user_input, top_k=4)

    if should_fallback(hits):
        # Fallback aktivieren
        st.session_state.fallback_mode = True
        st.session_state.fallback_data = {
            "original_question": user_input,
            "thema": "",
            "ziel": "",
            "frist": "",
            "kontext": "",
            "betreff": ""
        }
        st.session_state.messages.append({"role":"assistant",
            "content": "Ich finde dazu keine eindeutige Stelle in deinen Unterlagen. "
                       "Lass mich eine saubere Anfrage vorbereiten. "
                       "Dazu ein paar kurze Fragen: Thema? Ziel/Ergebnis? Frist/Termin? Optional kurzer Kontext."})
        st.rerun()

    # keine Fallback-Bedingung -> normal antworten + Quellen optional
    with st.spinner("Formuliere Antwort…"):
        out = answer_with_context(user_input, hits)
    follow = "\n\n—\nSoll ich dir zeigen, wo ich das gefunden habe? Antworte einfach mit **„ja“**."
    st.session_state.messages.append({"role":"assistant","content": out + follow})
    st.session_state.last_hits = hits
    st.session_state.await_snippet_confirm = True
    st.rerun()

# ---------- Fallback UI ----------
if st.session_state.get("fallback_mode", False):
    st.divider()
    st.subheader("📮 Fallback: Strukturierte Anfrage erstellen")
    with st.form("fallback_form"):
        st.session_state.fallback_data["thema"]   = st.text_input("Thema / Betreffzeile (kurz):", value=st.session_state.fallback_data.get("thema",""))
        st.session_state.fallback_data["ziel"]    = st.text_input("Gewünschtes Ergebnis / Entscheidung / Info:", value=st.session_state.fallback_data.get("ziel",""))
        st.session_state.fallback_data["frist"]   = st.text_input("Frist / Termin (optional):", value=st.session_state.fallback_data.get("frist",""))
        st.session_state.fallback_data["kontext"] = st.text_area("Kurzer Kontext (optional):", value=st.session_state.fallback_data.get("kontext",""), height=120)
        st.session_state.fallback_data["betreff"] = st.text_input("E-Mail Betreff (optional, sonst aus Thema):", value=st.session_state.fallback_data.get("betreff",""))

        st.markdown("**Empfänger aus den Unterlagen (automatisch erkannt):**")
        options = [f"{c['email']} — {c['label']} (Quelle: {c['source']} S.{c.get('page','?')})" for c in CONTACTS] or ["(keine Kontakte in Dokumenten gefunden)"]
        choice = st.selectbox("Vorschläge:", options, index=0)
        to_manual = st.text_input("Oder Empfänger manuell eintragen (E-Mail):", value="")

        submitted = st.form_submit_button("✍️ Anfrage-Entwurf erzeugen")
        if submitted:
            details = {
                "thema":   st.session_state.fallback_data["thema"]   or "Rückfrage",
                "ziel":    st.session_state.fallback_data["ziel"]    or "Bitte um Klärung",
                "frist":   st.session_state.fallback_data["frist"],
                "kontext": st.session_state.fallback_data["kontext"],
                "betreff": st.session_state.fallback_data["betreff"],
                "anrede":  "Hallo,"
            }
            draft = make_email_draft(st.session_state.fallback_data.get("original_question",""), details)
            st.success("Entwurf erstellt")
            st.write(f"**Betreff:** {draft['subject']}")
            st.text_area("E-Mail-Text", draft["body"], height=260)

            # Empfänger wählen
            picked_email = ""
            if CONTACTS and "keine Kontakte" not in options[0] and choice and "—" in choice:
                picked_email = choice.split(" — ")[0].strip()
            if to_manual.strip():
                picked_email = to_manual.strip()

            if picked_email:
                mailto_url = build_mailto(
                    to=picked_email,
                    subject=draft["subject"],
                    body=draft["body"]
                )
                st.markdown(f"[📧 Im eigenen Mailprogramm öffnen]({mailto_url})")
                with st.expander("Link anzeigen/kopieren"):
                    st.code(mailto_url, language="text")
            else:
                st.warning("Bitte einen Empfänger auswählen oder manuell eintragen.")

    if st.button("❌ Fallback schließen"):
        st.session_state.fallback_mode = False
        st.experimental_rerun()
