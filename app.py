# app.py
import os, json
import numpy as np
import streamlit as st
from datetime import datetime
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# --- Env Variablen / Secrets ---
def env(name, default=None):
    return st.secrets.get(name, os.getenv(name, default))

AZURE_OPENAI_KEY = env("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = env("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_CHAT_DEPLOYMENT = env("AZURE_OPENAI_CHAT_DEPLOYMENT")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = env("AZURE_OPENAI_API_VERSION", "2024-06-01")

if not AZURE_OPENAI_KEY or not AZURE_OPENAI_ENDPOINT:
    st.stop()

INDEX_DIR = "agentiva_db"
INDEX_NPZ  = os.path.join(INDEX_DIR, "index.npz")
INDEX_META = os.path.join(INDEX_DIR, "metadaten.json")
INDEX_INFO = os.path.join(INDEX_DIR, "index_info.json")
UNTERLAGEN_DIR = "unterlagen"

st.set_page_config(page_title="Agentiva – Wissenslotse", page_icon="🧭", layout="wide")
st.title("🤖 Agentiva – Ihr Wissenslotse")
st.caption("Antwortet faktenbasiert aus den PDFs im Ordner ‘unterlagen’. und schlägt Ansprechpartner vor, wenn Antworten nicht eindeutig sind.")

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

# --- Manifest-Funktionen ---
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
    saved = (info or {}).get("files", {})
    return saved != current

# --- Index laden/aufbauen ---
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
        for m in M:
            m.setdefault("last_modified", None)
        return E, M, info
    return None, None, {}

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
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
        except Exception:
            text = ""
        if not text.strip():
            continue
        try:
            mtime = os.path.getmtime(p)
        except Exception:
            mtime = None

        chunks = chunk_text(text, size=1200, overlap=200)[:300]
        for i, ch in enumerate(chunks):
            all_chunks.append(ch)
            meta.append({
                "source": os.path.basename(p),
                "chunk_id": i,
                "text": ch,
                "last_modified": mtime,
            })

    # Embeddings in Batches
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

    current = file_manifest(UNTERLAGEN_DIR)
    with open(INDEX_INFO, "w", encoding="utf-8") as f:
        json.dump({
            "built_at": datetime.now().isoformat(),
            "files": current
        }, f)

    return E, meta

# --- Index laden & Auto-Reindex ---
E, META, INFO = load_index()
current = file_manifest(UNTERLAGEN_DIR)
if (E is None) or is_index_stale(INFO, current):
    with st.spinner("Änderungen in 'unterlagen/' erkannt – baue Index neu…"):
        E, META = build_index_now()
        _, _, INFO = load_index()

# --- Sidebar ---
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
        st.session_state.messages = []
        st.rerun()

# --- Suche & Antwort ---
def retrieve(query: str, top_k: int = 4):
    if E is None or META is None:
        return []
    q = emb.embed_query(query)
    q = np.array(q, dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-10)
    sims = (E @ q)
    idx = np.argsort(-sims)[:top_k]
    return [{"score": float(sims[i]), **META[i]} for i in idx]

def answer_with_context(question: str, passages: list[dict]) -> str:
    context_blocks = []
    for p in passages:
        context_blocks.append(f"[{p['source']} • Abschnitt {p['chunk_id']}] {p['text'][:1200]}")
    context = "\n\n".join(context_blocks)

    system = (
        "Du bist Agentiva, ein Wissenslotse: ein KI-Assistent, der Anwendern hilft, Antworten in komplexen Dokumenten zu finden. "
        "Antworte stets faktenbasiert und klar in deutscher Sprache. "
        "Dein Ziel ist es, dem Nutzer so gut wie möglich eine finale, hilfreiche Antwort zu geben. "
        "Wenn die Antwort nicht eindeutig aus den bereitgestellten Kontexten hervorgeht, prüfe, ob im Kontext Ansprechpartner, "
        "Kontaktdaten oder Zuständigkeiten genannt sind. "
        "Falls ja, schlage diese als Ansprechpartner vor. "
        "Wenn keine Ansprechpartner genannt sind, sage ehrlich, dass im Dokument keine passenden Ansprechpartner gefunden wurden."
    )

    user_msg = f"Frage:\n{question}\n\nKontexte:\n{context}"

    msg = llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": user_msg}])
    return msg.content

# --- Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

st.subheader("💬 Chat")

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"👤 **Du:** {msg['content']}")
    else:
        st.markdown(f"🤖 **Agentiva:** {msg['content']}")
        if msg.get("sources"):
            with st.expander("📚 Quellen"):
                built_at = INFO.get("built_at")
                if built_at:
                    try:
                        dt = datetime.fromisoformat(built_at).strftime("%Y-%m-%d %H:%M")
                        st.caption(f"📅 Letzte Indexierung: {dt}")
                    except Exception:
                        st.caption(f"📅 Letzte Indexierung: {built_at}")
                for h in msg["sources"]:
                    when = ""
                    if h.get("last_modified"):
                        try:
                            doc_dt = datetime.fromtimestamp(h["last_modified"]).strftime("%Y-%m-%d")
                            when = f", geändert: {doc_dt}"
                        except Exception:
                            pass
                    st.markdown(
                        f"**{h['source']}** (Abschnitt {h['chunk_id']}, Score {h['score']:.3f}{when})"
                    )

user_input = st.chat_input("Schreibe deine Frage...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    if E is None:
        st.warning("Kein Index gefunden.")
    else:
        with st.spinner("Suche relevante Passagen…"):
            hits = retrieve(user_input, top_k=4)

        if not hits:
            answer = "Keine Treffer in der Wissensbasis. Im Dokument wurden keine Ansprechpartner gefunden."
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": []})
        else:
            with st.spinner("Formuliere Antwort…"):
                out = answer_with_context(user_input, hits)
            st.session_state.messages.append({"role": "assistant", "content": out, "sources": hits})

    st.rerun()
