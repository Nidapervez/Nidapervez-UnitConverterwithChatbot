import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.indexes import VectorstoreIndexCreator

# Unit conversion factors
conversion_factors = {
    "Plane Angle": {
        "Degree": 1.0,
        "Arcsecond": 1 / 3600,
        "Gradian": 0.9,
        "Milliradian": 0.0572958,
        "Minute of arc": 1 / 60,
        "Radian": 57.2958
    },
    "Length": {
        "Meter": 1.0,
        "Kilometer": 1000.0,
        "Centimeter": 0.01,
        "Millimeter": 0.001,
        "Micrometer": 0.000001,
        "Nanometer": 0.000000001,
        "Mile": 1609.34,
        "Yard": 0.9144,
        "Foot": 0.3048,
        "Inch": 0.0254,
        "Nautical Mile": 1852.0,
        "Fathom": 1.8288,
        "Furlong": 201.168,
        "Light Year": 9.461e+15
    },
    "Temperature": {
        "Celsius": {"to_base": lambda x: x, "from_base": lambda x: x},
        "Fahrenheit": {"to_base": lambda x: (x - 32) * 5/9, "from_base": lambda x: (x * 9/5) + 32},
        "Kelvin": {"to_base": lambda x: x - 273.15, "from_base": lambda x: x + 273.15}
    }
}

def convert_units(value, from_unit, to_unit, category):
    if category not in conversion_factors:
        return "Invalid category"
    
    factors = conversion_factors[category]
    if category == "Temperature":
        if from_unit not in factors or to_unit not in factors:
            return "Invalid unit"
        base_value = factors[from_unit]["to_base"](value)
        return factors[to_unit]["from_base"](base_value)
    else:
        if from_unit not in factors or to_unit not in factors:
            return "Invalid unit"
        return (value * factors[from_unit]) / factors[to_unit]

# Streamlit UI
st.set_page_config(page_title="Unit Converter & Chatbot", layout="wide")
st.title("⚡ Advanced Unit Converter & Chatbot")

col1, col2 = st.columns(2)

with col1:
    st.header("Unit Converter")
    conversion_type = st.selectbox("Select Conversion Type", list(conversion_factors.keys()))
    
    col_a, col_b, col_c = st.columns([2, 1, 2])
    
    with col_a:
        from_unit = st.selectbox("From Unit", list(conversion_factors[conversion_type].keys()))
    with col_b:
        st.markdown("<h2 style='text-align: center;'>=</h2>", unsafe_allow_html=True)
    with col_c:
        to_unit = st.selectbox("To Unit", list(conversion_factors[conversion_type].keys()))
    
    value1 = st.number_input("Enter Value", value=1.0, step=0.1, format="%f")
    
    try:
        value2 = convert_units(value1, from_unit, to_unit, conversion_type)
        st.markdown(f"<h2 style='text-align: center; color: blue;'>Result: {value2:.4f}</h2>", unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f"<h2 style='text-align: center; color: red;'>Error: {str(e)}</h2>", unsafe_allow_html=True)

# Chatbot Section
with col2:
    st.header("Ask Chatbot (Unit Converter)")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    @st.cache_resource
    def get_vectorstore():
        file_path = "unit_converter_data.txt"
        loader = TextLoader(file_path)
        documents = loader.load()
        index = VectorstoreIndexCreator(
            embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2"),
            text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        ).from_documents(documents)
        return index.vectorstore

    prompt = st.chat_input("Ask something about unit conversion...")
    
    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load document")
            
            model = "llama3-8b-8192"
            groq_chat = ChatGroq(
                groq_api_key=os.environ.get("GROQ_API_KEY"),
                model_name=model
            )
            
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
            st.error(f"Error: {str(e)}")
