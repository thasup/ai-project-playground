import streamlit as st
from dotenv import load_dotenv
import os
import time
import json
import base64

# LangChain imports
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders.parsers.pdf import (
    PyMuPDFParser,
)
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationSummaryMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from PyPDF2 import PdfReader

# Load environment variables
load_dotenv()

def pdf_read(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n"], chunk_size=200, chunk_overlap=0)
    chunks = text_splitter.split_text(text)
    return chunks

def vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_db")

# Page config
st.set_page_config(
    page_title="Document Expert",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================
# Sidebar Configuration
# ========================
with st.sidebar:
    st.sidebar.header("⚙️ Configuration")
    with st.expander("LLM Settings", expanded=True):
            openai_api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
            google_api_key = st.text_input("Google API Key", type="password", value=os.getenv("GOOGLE_API_KEY", ""))
            model_provider = st.radio(
                "Select Model Provider",
                ["OpenAI", "Google"],
                help="Choose which AI model provider to use"
            )
            if model_provider == "OpenAI":
                model = st.selectbox(
                    "Select OpenAI Model",
                    ["gpt-4o-mini", "gpt-o1-mini", "gpt-3.5-turbo", "gpt-4-turbo-preview", "gpt-4-vision-preview"],
                    help="Choose the OpenAI model for chat"
                )
            else:
                model = st.selectbox(
                    "Select Google Model",
                    ["gemini-2.0-flash", "gemini-1.5-flash"],
                    help="Choose the Google model for chat"
                )

# Set API keys as environment variables
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["GOOGLE_API_KEY"] = google_api_key

# Initialize the LLM based on the chosen provider and provided API key
if model_provider == "OpenAI" and openai_api_key:
    llm = ChatOpenAI(
        model=model,
        openai_api_key=openai_api_key,
    )
elif model_provider == "Google" and google_api_key:
    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=google_api_key,
    )
else:
    st.error("🔑 Please enter the required API key in the sidebar.")
    st.stop()

# File upload for OKR-related documents
uploaded_files = st.sidebar.file_uploader(
    "Upload OKR-related documents", type=["pdf", "txt", "csv"], accept_multiple_files=True
)

# Process uploaded files and save to session state
if uploaded_files:
    st.sidebar.info("Processing uploaded documents...")
    all_text_chunks = []

    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            loader = PyMuPDFLoader(uploaded_file)
            raw_text = pdf_read(uploaded_file)
        elif uploaded_file.type == "text/plain":
            content = uploaded_file.read().decode("utf-8")
            loader = TextLoader(content)
            raw_text = content
        elif uploaded_file.type == "text/csv":
            content = uploaded_file.read().decode("utf-8")
            loader = CSVLoader(content)
            raw_text = content
        else:
            st.error("Unsupported file type.")
            st.stop()

        # Split document into chunks
        text_chunks = get_chunks(raw_text)
        all_text_chunks.extend(text_chunks)

    # Save all text chunks to session state
    st.session_state.text_chunks = all_text_chunks
    st.success("Documents processed and saved to session state!")

# A text area for customizing the system prompt
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = (
        "You are an expert personal OKR coach. "
        "Help the user plan, create, execute, track, and review their OKRs."
    )
user_defined_prompt = st.sidebar.text_area(
    "System Prompt", value=st.session_state.system_prompt,
    help="Customize the AI assistant's behavior."
)
if st.sidebar.button("Update System Prompt"):
    st.session_state.system_prompt = user_defined_prompt
    st.sidebar.success("System prompt updated!")

# ========================
# Document Indexing Section
# ========================
persist_directory = "embeddings_db"  # directory to persist vector DB
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Create/update Chroma vector store
db = Chroma.from_texts(st.session_state.text_chunks, embedding=embeddings)
db.persist()
st.sidebar.success("Document processed and indexed!")

# ========================
# Initialize Conversation Memory
# ========================
if "memory" not in st.session_state:
    st.session_state.memory = ConversationSummaryMemory(
        return_messages=True,
        memory_key="chat_history",
        llm=llm
    )
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm here to help you with your personal OKRs. What would you like to work on today?"}
    ]

# ========================
# Initialize LLM and QA Chain
# ========================
retriever = db.as_retriever(search_type="similarity", k=4)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, retriever=retriever, chain_type="stuff"
)

# ========================
# Main Chat Interface
# ========================
st.title("Personal OKRs Assistant")
st.markdown(
    "Use this tool to plan, create, execute, track, and review your personal OKRs. "
    "Ask questions like *'How should I prioritize my OKRs?'* or share your progress for feedback."
)

# Display conversation history
st.subheader("Conversation")
for msg in st.session_state.messages:
    if msg["role"] == "assistant":
        st.markdown("**Assistant:** " + msg["content"])
    else:
        st.markdown("**You:** " + msg["content"])

# Chat input widget
user_query = st.text_input("Enter your message:")
if st.button("Send") and user_query:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.session_state.memory.save_context({"content": user_query}, {"output": ""})

    # Determine how to respond:
    # If the query is related to OKRs and we have a document index, use the QA chain.
    if qa_chain is not None and any(word in user_query.upper() for word in ["OKR", "OBJECTIVE", "KEY RESULT", "PRIORITIZE"]):
        response = qa_chain.run(user_query)
    else:
        # Use conversation history with a chat prompt.
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", st.session_state.system_prompt + " Always use '\\n' for line breaks in your responses."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{content}")
        ])
        chain = prompt_template | llm | StrOutputParser()
        chat_history = st.session_state.memory.load_memory_variables({})["chat_history"]
        response = chain.invoke({"content": user_query, "chat_history": chat_history})

    # Simulate typing effect (optional)
    with st.spinner("Thinking..."):
        time.sleep(1)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.memory.save_context({"content": ""}, {"output": response})
    st.rerun()

# ========================
# OKR Management Section
# ========================
st.subheader("Manage Your Personal OKRs")
if "okrs" not in st.session_state:
    st.session_state.okrs = []
okr_title = st.text_input("OKR Title", key="okr_title")
okr_objective = st.text_area("Objective", key="okr_objective")
okr_key_results = st.text_area("Key Results (comma separated)", key="okr_key_results")
if st.button("Add OKR"):
    okr = {
        "title": okr_title,
        "objective": okr_objective,
        "key_results": [kr.strip() for kr in okr_key_results.split(",") if kr.strip()]
    }
    st.session_state.okrs.append(okr)
    st.success("OKR added!")

if st.session_state.okrs:
    st.markdown("### Your OKRs")
    for idx, okr in enumerate(st.session_state.okrs, start=1):
        st.markdown(f"**OKR {idx}: {okr['title']}**")
        st.markdown(f"*Objective:* {okr['objective']}")
        st.markdown(f"*Key Results:* {', '.join(okr['key_results'])}")
