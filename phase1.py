import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
  # Updated import
from langchain.indexes import VectorstoreIndexCreator

# Ensure API key is set
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    st.error("🚨 Error: GROQ API key is missing! Set it in the environment or pass it in the code.")
    st.stop()

# Streamlit UI Configuration
st.set_page_config(page_title="Neon Unit Converter & Chatbot", layout="wide")
st.markdown(
    """
    <style>
        body {
            background-color: black;
            color: white;
        }
        .stButton button {
            background-color: #00ffcc;
            color: black;
            font-weight: bold;
            border-radius: 10px;
        }
        .stSelectbox, .stNumber_input {
            border-radius: 10px;
        }
        .result-box {
            background: linear-gradient(135deg, #00ffcc, #ff00ff);
            padding: 15px;
            text-align: center;
            border-radius: 10px;
            color: black;
            font-weight: bold;
            font-size: 20px;
        }
        .chatbox {
            background: #111;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #00ffcc;
        }
        .header-text {
            color: #ff00ff;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Unit conversion factors
conversion_factors = {
    "Length": {
        "Meter": 1.0, "Kilometer": 1000.0, "Centimeter": 0.01, "Millimeter": 0.001,
        "Mile": 1609.34, "Yard": 0.9144, "Foot": 0.3048, "Inch": 0.0254
    },
    "Temperature": {
        "Celsius": {"to_base": lambda x: x, "from_base": lambda x: x},
        "Fahrenheit": {"to_base": lambda x: (x - 32) * 5/9, "from_base": lambda x: (x * 9/5) + 32},
        "Kelvin": {"to_base": lambda x: x - 273.15, "from_base": lambda x: x + 273.15}
    }
}

def convert_units(value, from_unit, to_unit, category):
    factors = conversion_factors[category]
    if category == "Temperature":
        base_value = factors[from_unit]["to_base"](value)
        return factors[to_unit]["from_base"](base_value)
    return (value * factors[from_unit]) / factors[to_unit]

# Unit Converter Section
st.markdown("<h1 class='header-text'>⚡ Neon Unit Converter</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid #00ffcc;'>", unsafe_allow_html=True)

conversion_type = st.selectbox("Select Conversion Type", list(conversion_factors.keys()))
col1, col2 = st.columns(2)

with col1:
    from_unit = st.selectbox("From Unit", list(conversion_factors[conversion_type].keys()))
    value1 = st.number_input("Enter Value", value=1.0, step=0.1, format="%f")

with col2:
    to_unit = st.selectbox("To Unit", list(conversion_factors[conversion_type].keys()))
    result = convert_units(value1, from_unit, to_unit, conversion_type)

st.markdown(f"<div class='result-box'>Result: {result:.4f} {to_unit}</div>", unsafe_allow_html=True)

# Chatbot Section
st.markdown("<h1 class='header-text'>🤖 AI Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid #ff00ff;'>", unsafe_allow_html=True)

@st.cache_resource
def get_vectorstore():
    loader = TextLoader("unit_converter_data.txt")
    documents = loader.load()
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2"),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    ).from_documents(documents)
    return index.vectorstore

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
