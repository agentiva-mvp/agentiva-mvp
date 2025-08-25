import os
import streamlit as st
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Chroma

# === Parameter ===
PERSIST_DIR = "agentiva_db"

ENDPOINT = "https://mirko-memqopyf-eastus2.services.ai.azure.com/"
API_VERSION = "2024-06-01"
CHAT_DEPLOYMENT = "gpt-4o-mini"
EMBEDDING_DEPLOYMENT = "embedding_small"

# === API-Keys laden ===
if not os.getenv("AZURE_OPENAI_KEY"):
    raise ValueError("❌ Kein API-Key gefunden! Bitte 'AZURE_OPENAI_KEY' als Umgebungsvariable setzen.")

# === Embeddings & LLM ===
embeddings = AzureOpenAIEmbeddings(
    deployment=EMBEDDING_DEPLOYMENT,
    openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=ENDPOINT,
    openai_api_version=API_VERSION
)

llm = AzureChatOpenAI(
    deployment_name=CHAT_DEPLOYMENT,
    openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=ENDPOINT,
    openai_api_version=API_VERSION,
    temperature=0.2
)

# === Vektor-DB laden ===
vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
qa = ConversationalRetrievalChain.from_llm(llm, vectordb.as_retriever())

# === Streamlit UI ===
st.title("🤖 Agentiva – KI-Agent für den Marketing-Dschungel")
st.write("Antwortet faktenbasiert aus den PDFs im Ordner **unterlagen**.")

if "history" not in st.session_state:
    st.session_state.history = []

user_question = st.text_input("❓ Stelle eine Frage an die Wissensdatenbank:")

if user_question:
    result = qa({"question": user_question, "chat_history": st.session_state.history})
    st.session_state.history.append((user_question, result["answer"]))
    st.write(f"**Antwort:** {result['answer']}")
