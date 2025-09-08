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
st.caption("Antwortet faktenbasiert aus PDFs. Wenn keine sichere Antwort möglich ist, formuliert Agentiva im Chat eine Anfrage und öffnet dein Mailprogramm vorbefüllt.")

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
if E is None:
    with st.spinner("Baue Wissensindex…"):
        E, META, CONTACTS = build_index_now()

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
FALLBACK_MIN_GOOD = int(env("FALLBACK_MIN_GOOD", "2"))

def should_fallback(hits: List[Dict[str, Any]]) -> bool:
    if not hits: return True
    good = sum(1 for h in hits if h["score"] >= FALLBACK_MIN_SIM)
    return good < FALLBACK_MIN_GOOD

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

# ---------- Chat UI ----------
st.subheader("💬 Chat")

for msg in st.session_state.messages:
    role = "Du" if msg["role"] == "user" else "Agentiva"
    icon = "👤" if msg["role"] == "user" else "🤖"
    st.markdown(f"{icon} **{role}:** {msg['content']}")

user_input = st.chat_input("Schreibe deine Frage…")

# ---------- Snippet-Handler ----------
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
if user_input and not st.session_state.await_snippet_confirm:
    st.session_state.messages.append({"role":"user","content":user_input})
    hits = pdf_hits(user_input, top_k=4)
    if should_fallback(hits):
        st.session_state.fallback_step = "ask_consent"
        st.session_state.fallback_data = {"original_question": user_input}
        st.session_state.messages.append({
            "role":"assistant",
            "content":"Ich finde keine eindeutige Antwort in den Dokumenten. "
                      "Soll ich dir helfen, eine Anfrage zu formulieren?"
        })
        st.rerun()
    else:
        strong = [h for h in hits if h["score"] >= FALLBACK_MIN_SIM]
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
