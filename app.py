# app.py
import os, json, unicodedata
import numpy as np
import streamlit as st
from urllib.parse import quote, quote_plus
from typing import List, Dict, Any
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# --- Helpers ---
def env(name, default=None):
    return st.secrets.get(name, os.getenv(name, default))

# Normalisierung für robuste Eingabeerkennung
def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return s

YES_TOKENS = {
    "ja","j","yes","y","ok","okay","okey","gern","gerne","passt","passt so","klingt gut",
    "mach","mach mal","ist ok","ist okay","alles klar","go","bitte","ja bitte","sounds good"
}
NO_TOKENS = {
    "nein","n","no","nope","lass","lass sein","abbrechen","spater","später","nicht","ne","nee"
}
FALLBACK_HINT_TOKENS = {
    "hilfe","hilf","unterstutze","unterstütze","anfrage","mail schreiben","mail verfassen",
    "formuliere","kontakt aufnehmen","ansprechpartner","zustandiger","zuständiger"
}

def is_yes(text: str) -> bool:
    t = _norm(text)
    return any(tok in t for tok in YES_TOKENS)

def is_no(text: str) -> bool:
    t = _norm(text)
    return any(tok in t for tok in NO_TOKENS)

def wants_fallback(text: str) -> bool:
    t = _norm(text)
    return any(tok in t for tok in FALLBACK_HINT_TOKENS)

# --- Azure Config ---
AZURE_OPENAI_KEY = env("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = env("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_CHAT_DEPLOYMENT = env("AZURE_OPENAI_CHAT_DEPLOYMENT")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = env("AZURE_OPENAI_API_VERSION", "2024-06-01")

if not AZURE_OPENAI_KEY or not AZURE_OPENAI_ENDPOINT:
    st.stop()

# --- Pfade ---
INDEX_DIR = "agentiva_db"
INDEX_NPZ = os.path.join(INDEX_DIR, "index.npz")
INDEX_META = os.path.join(INDEX_DIR, "metadaten.json")
UNTERLAGEN_DIR = "unterlagen"

# --- GitHub Config für Deeplinks ---
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

# --- Azure Clients ---
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

# --- Index Laden/Aufbauen ---
def load_index() -> tuple[np.ndarray | None, list[dict] | None]:
    if os.path.exists(INDEX_NPZ) and os.path.exists(INDEX_META):
        try:
            data = np.load(INDEX_NPZ)
            E = data["embeddings"].astype(np.float32)
            with open(INDEX_META, "r", encoding="utf-8") as f:
                M = json.load(f)
            # Robustheits-Check
            if not M or not all("source" in m and "text" in m for m in M):
                return None, None
            if not all("page" in m for m in M):  # fehlende Seiten → neu bauen
                return None, None
            return E, M
        except Exception:
            return None, None
    return None, None

def build_index_now():
    from pypdf import PdfReader

    def chunk_text(text: str, size: int = 1200, overlap: int = 200):
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

    all_chunks, meta = [], []
    for p in pdfs:
        try:
            reader = PdfReader(p)
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if not text.strip():
                    continue
                chunks = chunk_text(text)[:50]
                for i, ch in enumerate(chunks):
                    all_chunks.append(ch)
                    meta.append({"source": os.path.basename(p), "chunk_id": i, "page": page_num, "text": ch})
        except Exception:
            continue

    vecs = []
    BATCH = 64
    for i in range(0, len(all_chunks), BATCH):
        batch = all_chunks[i:i+BATCH]
        vecs.extend(emb.embed_documents(batch))
    E = np.array(vecs, dtype=np.float32)
    E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-10)

    os.makedirs(INDEX_DIR, exist_ok=True)
    np.savez_compressed(INDEX_NPZ, embeddings=E)
    with open(INDEX_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return E, meta

E, META = load_index()
if E is None or META is None:
    with st.spinner("Baue Wissensindex neu auf…"):
        E, META = build_index_now()

# --- Streamlit UI ---
st.set_page_config(page_title="Agentiva – Ihr Wissenslotse", page_icon="🧭", layout="wide")
st.title("🤖 Agentiva – Ihr Wissenslotse")
st.caption("Antwortet faktenbasiert aus den hinterlegten PDFs. Mit Fallback-Support bei unklaren Fragen.")

# Sidebar
with st.sidebar:
    st.subheader("⚙️ Index-Verwaltung")
    if st.button("🔄 Index jetzt neu bauen"):
        with st.spinner("Baue Wissensindex…"):
            E, META = build_index_now()
            if E is not None:
                st.success(f"Index gebaut: {E.shape[0]} Chunks")
            else:
                st.error("Index konnte nicht gebaut werden.")
    if E is not None:
        st.info(f"Aktiver Index: {E.shape[0]} Chunks")

# --- Retrieval ---
def retrieve(query: str, top_k: int = 4):
    if E is None or META is None:
        return []
    q = emb.embed_query(query)
    q = np.array(q, dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-10)
    sims = (E @ q)
    idx = np.argsort(-sims)[:top_k]
    results = [{"score": float(sims[i]), **META[i]} for i in idx]
    return results

FALLBACK_MIN_SIM = float(env("FALLBACK_MIN_SIM", "0.30"))
FALLBACK_MIN_GOOD = int(env("FALLBACK_MIN_GOOD", "1"))

def should_fallback(hits: List[Dict[str, Any]]) -> bool:
    if not hits:
        return True
    good = [h for h in hits if h.get("score", 0) >= FALLBACK_MIN_SIM]
    return len(good) < FALLBACK_MIN_GOOD

# --- Session State ---
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
if "deferred_user_input" not in st.session_state:
    st.session_state.deferred_user_input = None

# --- Chat UI ---
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_input = st.chat_input("Stelle deine Frage…")

# Vorrang für deferred input
if st.session_state.deferred_user_input and not user_input:
    user_input = st.session_state.deferred_user_input
    st.session_state.deferred_user_input = None

if user_input:
    st.session_state.messages.append({"role":"user","content":user_input})

    # Expliziter Fallbackwunsch
    if wants_fallback(user_input):
        st.session_state.fallback_step = "ask_thema"
        st.session_state.fallback_data = {"original_question": user_input}
        st.session_state.messages.append({"role":"assistant","content":"Alles klar. Worum geht es genau (Thema)?"})
        st.rerun()

    # Snippet-Handler
    if st.session_state.await_snippet_confirm:
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
            st.session_state.await_snippet_confirm = False
            st.session_state.deferred_user_input = user_input
            st.rerun()

    # Normale Retrieval-Antwort
    hits = retrieve(user_input, top_k=4)
    if should_fallback(hits):
        st.session_state.fallback_step = "ask_consent"
        st.session_state.fallback_data = {"original_question": user_input}
        st.session_state.messages.append({"role":"assistant","content":"Ich finde keine eindeutige Antwort. Soll ich dir helfen, eine Anfrage zu formulieren?"})
        st.rerun()
    else:
        strong = [h for h in hits if h["score"] >= FALLBACK_MIN_SIM]
        blocks = "\n\n".join([
            f"[{h.get('source','?')} • Seite {h.get('page','?')}] {(h.get('text','') or '')[:300]}"
            for h in strong
        ])
        system = "Du bist ein Assistent, der faktenbasiert aus Dokumenten antwortet."
        user_msg = f"Frage:\n{user_input}\n\nKontexte:\n{blocks}"
        msg = llm.invoke([{"role":"system","content":system},{"role":"user","content":user_msg}])
        st.session_state.messages.append({"role":"assistant","content":msg.content})
        st.session_state.messages.append({
            "role":"assistant",
            "content":"Möchtest du die entsprechenden Stellen im Original sehen? Antworte mit **ja**.",
        })
        st.session_state.await_snippet_confirm = True
        st.session_state.last_hits = strong
        st.rerun()
