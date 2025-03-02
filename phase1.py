import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain.indexes import VectorstoreIndexCreator

# Ensure API key is set
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    st.error("🚨 Error: GROQ API key is missing! Set it in the environment or pass it in the code.")
    st.stop()

@st.cache_resource  # Updated syntax
def get_vectorstore():
    loader = TextLoader("unit_converter_data.txt")
    documents = loader.load()
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2"),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    ).from_documents(documents)
    return index.vectorstore

st.markdown("<h1 class='header-text'>🤖 AI Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid #ff00ff;'>", unsafe_allow_html=True)

st.markdown("<div class='chatbox'>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

prompt = st.chat_input("Ask me anything about unit conversion...")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        vectorstore = get_vectorstore()
        groq_chat = ChatGroq(groq_api_key=API_KEY, model_name="llama3-8b-8192")

        chain = RetrievalQA.from_chain_type(
            llm=groq_chat,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )

        result = chain({"query": prompt})
        response = result["result"]

        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"🚨 Error: {str(e)}")

st.markdown("</div>", unsafe_allow_html=True)
