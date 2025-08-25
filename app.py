# app.py
import os, json
import numpy as np
import streamlit as st
from datetime import datetime
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# --- Env Variablen / Secrets ---
def env(name, default=None):
    # Streamlit Cloud: st.secrets bevorzugen, sonst os.getenv
    return st.secrets.get(name, os.getenv(name, default))

AZURE_OPENAI_KEY = env("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = env("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_CHAT_DEPLOYMENT = env("AZURE_OPENAI_CHAT_DEPLOYMENT")        # z.B. "gpt-4o-mini"
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")  # z.B. "embedding_small"
AZURE_OPENAI_API_VERSION = env("AZURE_OPENAI_API_VERSION", "2024-06-01")

if not AZURE_OPENAI_KEY or not AZURE_OPENAI_ENDPOINT:
    st.stop()

INDEX_DIR = "agentiva_db"
INDEX_NPZ = os.path.join(INDEX_DIR, "index.npz")
INDEX_META = os.path.join(INDEX_DIR, "metadaten.json")
INDEX_INFO = os.path.join(INDEX_DIR, "index_info.json")  # extra für Index-Zeitpunkt
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
        info = {}
        if os.path.exists(INDEX_INFO):
            try:
                with open(INDEX_INFO, "r", encoding="utf-8") as f:
                    info = json.load(f)
            except Exception:
                info = {}
        # Fallback: alte Indexe ohne last_modified ergänzen
        for m in M:
            m.setdefault("last_modified", None)
        return E, M, info
    return None, None, {}

def build_index_now():
    from pypdf import PdfReader

    def chunk_text(text: str, size: int = 1200, overlap: int = 200):
        text = " ".join(text.split())
        chunks = []
        start = 0
        n = len(text)
        while start < n:
            end = min(start + size, n)
            chunks.append(text[start:end])
            if end == n:
                break
            start = end - overlap
            if start < 0:
                start = 0
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

        # Änderungszeitpunkt der Datei erfassen
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

    # Index-Bau-Datum speichern
    with open(INDEX_INFO, "w", encoding="utf-8") as f:
        json.dump({"built_at": datetime.now().isoformat()}, f)

    return E, meta

E, META, INFO = load_index()

with st.sidebar:
    st.subheader("⚙️ Index-Verwaltung")
    if st.button("🔄 Index jetzt neu bauen"):
        with st.spinner("Baue Wissensindex…"):
            E, META = build_index_now()
            # Info neu laden
            _, _, INFO = load_index()
            if E is not None:
                st.success(f"Index gebaut: {E.shape[0]} Chunks")
            else:
                st.error("Index konnte nicht gebaut werden.")
    if E is not None:
        st.info(f"Aktiver Index: {E.shape[0]} Chunks")
    built_at = INFO.get("built_at")
    if built_at:
        try:
            dt = datetime.fromisoformat(built_at).strftime("%Y-%m-%d %H:%M")
            st.caption(f"📅 Index zuletzt aktualisiert: {dt}")
        except Exception:
            st.caption(f"📅 Index zuletzt aktualisiert: {built_at}")

# --- Suche & Antwort ---
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

def answer_with_context(question: str, passages: list[dict]) -> str:
    context_blocks = []
    for p in passages:
        context_blocks.append(
            f"[{p['source']} • Abschnitt {p['chunk_id']}] {p['text'][:1200]}"
        )
    context = "\n\n".join(context_blocks)
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
                built_at = INFO.get("built_at")
                if built_at:
                    try:
                        dt = datetime.fromisoformat(built_at).strftime("%Y-%m-%d %H:%M")
                        st.caption(f"📅 Letzte Indexierung: {dt}")
                    except Exception:
                        st.caption(f"📅 Letzte Indexierung: {built_at}")
                for h in hits:
                    # Änderungsdatum pro Dokument
                    if h.get("last_modified"):
                        try:
                            doc_dt = datetime.fromtimestamp(h["last_modified"]).strftime("%Y-%m-%d")
                            when = f", geändert: {doc_dt}"
                        except Exception:
                            when = ""
                    else:
                        when = ""
                    st.markdown(
                        f"**{h['source']}** (Abschnitt {h['chunk_id']}, Score {h['score']:.3f}{when})"
                    )
            with st.spinner("Formuliere Antwort…"):
                out = answer_with_context(frage, hits)
            st.success("Antwort:")
            st.write(out)
            st.caption("Antwort generiert mit Azure OpenAI (kontextbasiert aus deinen PDFs).")
