import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayHnswSearch

# === Einstellungen ===
DATA_DIR = "unterlagen"       # Ordner mit deinen PDFs
PERSIST_DIR = "agentiva_db"   # Speicherort für die Vektordatenbank
DEPLOYMENT_NAME = "embedding_small"  # dein Embedding-Deployment
API_VERSION = "2024-06-01"    # oder die Version aus Azure
ENDPOINT = "https://mirko-memqopyf-eastus2.services.ai.azure.com/models"

# API-Key sicher aus Umgebungsvariable laden
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
if not AZURE_OPENAI_KEY:
    raise ValueError("❌ Kein API-Key gefunden! Bitte export AZURE_OPENAI_KEY setzen.")

# === Embeddings initialisieren ===
embeddings = OpenAIEmbeddings(
    model=DEPLOYMENT_NAME,
    openai_api_key=AZURE_OPENAI_KEY,
    openai_api_version=API_VERSION,
    openai_api_base=ENDPOINT
)

# === Hauptfunktion ===
def main():
    print("📚 Lade Dokumente…")
    docs = []
    for file in os.listdir(DATA_DIR):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_DIR, file))
            docs.extend(loader.load())

    print(f"➡️ {len(docs)} Dokumente gefunden")

    # Texte in Chunks teilen
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"➡️ In {len(chunks)} Text-Chunks geteilt")

    # Vektordatenbank aufbauen
    vectordb = DocArrayHnswSearch.from_documents(
        chunks,
        embeddings,
        work_dir=PERSIST_DIR,
        n_dim=1536  # Dimension für Embedding-Small (1536 für text-embedding-3-small)
    )

    print("✅ Wissensdatenbank erstellt und gespeichert!")

if __name__ == "__main__":
    main()
