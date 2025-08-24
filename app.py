import os
import streamlit as st
from langchain_openai import AzureChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayHnswSearch
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# === Einstellungen ===
PERSIST_DIR = "agentiva_db"   # Ordner mit der Datenbank
DEPLOYMENT_NAME = "gpt-4o"    # dein Chat-Deployment
EMBEDDING_NAME = "embedding_small"  # dein Embedding-Deployment
API_VERSION = "2024-06-01"    # Version aus Azure
ENDPOINT = "https://mirko-memqopyf-eastus2.services.ai.azure.com/models"

# API-Key aus Umgebungsvariable
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
if not AZURE_OPENAI_KEY:
    raise ValueError("❌ Kein API-Key gefunden! Bitte export AZURE_OPENAI_KEY setzen.")

# === Embeddings und LLM initialisieren ===
embeddings = OpenAIEmbeddings(
    model=EMBEDDING_NAME,
    openai_api_key=AZURE_OPENAI_KEY,
    openai_api_version=API_VERSION,
    openai_api_base=ENDPOINT
)

vectordb = None
if os.path.isdir(PERSIST_DIR):
    vectordb = DocArrayHnswSearch(
        embeddings,
        work_dir=PERSIST_DIR,
        n_dim=1536  # Dimension für embedding-small
    )

llm = AzureChatOpenAI(
    deployment_name=DEPLOYMENT_NAME,
    openai_api_key=AZURE_OPENAI_KEY,
    openai_api_base=ENDPOINT,
    openai_api_version=API_VERSION,
    temperature=0.2
)

# === Retrieval-Chain mit Memory ===
if vectordb:
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectordb.as_retriever(),
        memory=memory
    )
else:
    qa = None

# === Streamlit UI ===
st.set_page_config(page_title="Agentiva – KI-Agent für Vertrieb", page_icon="🤖")

st.title("🤖 Agentiva – Dein Vertriebs-Assistent")
st.write("Stelle Fragen zu den Dokumenten deines Unternehmens (Marketing, Richtlinien, Tools).")

if qa:
    user_question = st.text_input("❓ Deine Frage eingeben:")
    if user_question:
        with st.spinner("Denke nach..."):
            result = qa({"question": user_question})
