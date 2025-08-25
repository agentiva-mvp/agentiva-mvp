import os
import glob
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

# === Parameter ===
PERSIST_DIR = "agentiva_db"
DOCS_DIR = "unterlagen"

ENDPOINT = "https://mirko-memqopyf-eastus2.services.ai.azure.com/"
API_VERSION = "2024-06-01"
EMBEDDING_DEPLOYMENT = "embedding_small"

def load_documents():
    docs = []
    for file in glob.glob(f"{DOCS_DIR}/*.pdf"):
        loader = PyPDFLoader(file)
        docs.extend(loader.load())
    return docs

def main():
    # Embeddings für Azure
    embeddings = AzureOpenAIEmbeddings(
        deployment=EMBEDDING_DEPLOYMENT,
        openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
        azure_endpoint=ENDPOINT,
        openai_api_version=API_VERSION
    )

    # PDFs laden und in Chunks teilen
    docs = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    # Vektor-DB bauen & speichern
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIR)
    vectordb.persist()
    print(f"✅ Wissensdatenbank erstellt: {len(docs)} Dokumente, {len(chunks)} Chunks")

if __name__ == "__main__":
    main()
