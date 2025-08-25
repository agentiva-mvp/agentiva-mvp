# app.py
import os, json
import numpy as np
import streamlit as st
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# --- Helper für Variablen ---
def env(name, default=None):
    """Hole Variablen zuerst aus st.secrets, sonst aus Umgebungsvariablen"""
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name, default)

# --- Azure Konfiguration ---
AZURE_OPENAI_KEY = env("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = env("AZURE_OPENAI_ENDPOINT")  # z.B. "https://<dein-resource-name>.openai.azure.com/"
AZURE_OPENAI_CHAT_DEPLOYMENT = env("AZURE_OPENAI_CHAT_DEPLOYMENT")  # z.B. "gpt-4o-mini"
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")  # z.B. "embedding-small"
AZURE_OPENAI_API_VERSION = env("AZURE_OPENAI_API_VERSION", "2024-06-01")

if not AZURE_OPENAI_KEY or not AZURE_OPENAI_ENDPOINT:
    st.error("❌ Azure OpenAI Key oder Endpoint fehlen – bitte in den Streamlit Secrets setzen.")
    st.stop()

# --- Index-Dateien ---
INDEX_DIR = "agentiva_db"
INDEX_NPZ = os.path.join(INDEX_DIR, "index.npz")
INDEX_META = os.path.join(INDEX_DIR, "metadaten.json")
UNTERLAGEN_DIR = "unterlagen"

st.set_page_config(page_title="Agentiva – Marketing-Lotse", page_icon="🧭", layout="wide")
st.title("🤖 Agentiva – KI-Agent für den Marketing-Dschungel")
st.caption("Antwortet faktenbasiert aus den PDFs im Ordner ‘unterlagen’. (NumPy-Index, Cloud-freundlich)")

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

# --- Index laden/aufbauen ---
def load_index():
    if os.path.exists(INDEX_NPZ) and os.path.exists(INDEX_META):
        data = np.load(INDEX_NPZ)
        E = data["embeddings"].astype(np.float32)
        with open(INDEX_META, "r", encoding="utf-8") as f:
            M = json.load(f)
        return E, M
    return None, None

def build_index_now():
    from pypdf import PdfReader

    def chunk_text(text: str, size: int = 1200, overlap: int = 200):
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
        chunks = chunk_text(text, size=1200, overlap=200)[:300]
        for i, ch in enumerate(chunks):
            all_chunks.append(ch)
            meta.append({"source": os.path.basename(p), "chunk_id": i, "text": ch})

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
    return E, meta

E, META = load_index()

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

# --- Suche & Antwort ---
def retrieve(query: str, top_k: int = 4):
    if E is None or META is None:
        return []
    q = emb.embed_query(query)
    q = np.array(q, dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-10)
    sims = (E @ q)  # cosine similarity
    idx = np.argsort(-sims)[:top_k]
    return [{"score": float(sims[i]), **META[i]} for i in idx]

def answer_with_context(question: str, passages: list[dict]) -> str:
    context = "\n\n".join(
        f"[{p['source']} • Abschnitt {p['chunk_id']}] {p['text'][:1200]}" for p in passages
    )
    system = (
        "Du bist ein Assistent für Marketing- und Vertriebsunterlagen in Versicherungen. "
        "Antworte ausschließlich auf Basis der bereitgestellten Kontexte. "
        "Wenn die Antwort nicht sicher aus den Quellen hervorgeht, sage ehrlich, dass es nicht eindeutig ist."
    )
    user_msg = f"Frage:\n{question}\n\nKontexte:\n{context}"
    msg = llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": user_msg}])
    return msg.content

st.divider()
frage = st.text_input("🔎 Deine Frage an die Wissensbasis")
if st.button("Antwort holen") and frage.strip():
    if E is None:
        st.warning("Kein Index gefunden. Baue ihn in der Sidebar oder führe lokal `python build_db.py` aus.")
    else:
        with st.spinner("Suche relevante Passagen…"):
            hits = retrieve(frage, top_k=4)
        if not hits:
            st.info("Keine Treffer in der Wissensbasis.")
        else:
            with st.expander("Gefundene Passagen / Quellen"):
                for h in hits:
                    st.markdown(f"**{h['source']}** (Abschnitt {h['chunk_id']}, Score {h['score']:.3f})")
            with st.spinner("Formuliere Antwort…"):
                out = answer_with_context(frage, hits)
            st.success("Antwort:")
            st.write(out)
            st.caption("Antwort generiert mit Azure OpenAI (kontextbasiert aus deinen PDFs).")
