# build_db.py
import os, json, glob
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from pypdf import PdfReader
from langchain_openai import AzureOpenAIEmbeddings

# --- Umgebungsvariablen ---
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")  # z.B. "embedding_small"
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")

if not AZURE_OPENAI_KEY or not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_EMBEDDING_DEPLOYMENT:
    raise ValueError("Fehlende Env Vars: AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

UNTERLAGEN_DIR = "unterlagen"
OUT_DIR = "agentiva_db"
os.makedirs(OUT_DIR, exist_ok=True)

# --- einfache Chunk-Funktion ---
def chunk_text(text: str, size: int = 1200, overlap: int = 200):
    text = " ".join(text.split())  # normalisieren
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

def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    return "\n".join(parts)

def main():
    pdf_paths = sorted(glob.glob(os.path.join(UNTERLAGEN_DIR, "*.pdf")))
    if not pdf_paths:
        raise FileNotFoundError(f"Keine PDFs in '{UNTERLAGEN_DIR}/' gefunden.")

    # 1) PDFs -> Texte
    docs = []
    for p in pdf_paths:
        txt = extract_text_from_pdf(p)
        if txt.strip():
            docs.append({"source": os.path.basename(p), "text": txt})

    # 2) Chunking
    all_chunks = []
    metadaten = []
    for d in docs:
        chunks = chunk_text(d["text"], size=1200, overlap=200)
        for i, ch in enumerate(chunks):
            all_chunks.append(ch)
            metadaten.append({"source": d["source"], "chunk_id": i, "text": ch})

    # 3) Embeddings-Client
    embeddings_client = AzureOpenAIEmbeddings(
        azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
    )

    # 4) Embeddings in Batches
    vectors = []
    BATCH = 64
    for i in range(0, len(all_chunks), BATCH):
        batch = all_chunks[i:i+BATCH]
        vecs = embeddings_client.embed_documents(batch)
        vectors.extend(vecs)

    vectors_np = np.array(vectors, dtype=np.float32)  # (N, d)
    norms = np.linalg.norm(vectors_np, axis=1, keepdims=True) + 1e-10
    vectors_np = vectors_np / norms  # normalisieren für Cosine

    # 5) Speichern
    np.savez_compressed(os.path.join(OUT_DIR, "index.npz"), embeddings=vectors_np)
    with open(os.path.join(OUT_DIR, "metadaten.json"), "w", encoding="utf-8") as f:
        json.dump(metadaten, f, ensure_ascii=False, indent=2)

    print(f"✅ Index gebaut: {vectors_np.shape[0]} Chunks, Dimension {vectors_np.shape[1]}")
    print(f"   Ordner: {OUT_DIR}/ (index.npz, metadaten.json)")

if __name__ == "__main__":
    main()
