import streamlit as st
from dotenv import load_dotenv
import os
import time
import json
import base64

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationSummaryMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

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
uploaded_file = st.sidebar.file_uploader(
    "Upload an OKR-related document", type=["pdf", "txt", "csv"]
)

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

if uploaded_file is not None and uploaded_file.size > 0:
    st.sidebar.info("Processing uploaded document...")
    # Read the contents of the uploaded file
    if uploaded_file.type == "application/pdf":
        loader = PyPDFLoader(uploaded_file)
    elif uploaded_file.type == "text/plain":
        content = uploaded_file.read().decode("utf-8")
        loader = TextLoader(content)
    elif uploaded_file.type == "text/csv":
        content = uploaded_file.read().decode("utf-8")
        loader = CSVLoader(content)
    else:
        st.error("Unsupported file type.")
        st.stop()

    st.write(f"Processing file: {uploaded_file.name}, Type: {uploaded_file.type}")

    # Split document into chunks
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=200, chunk_overlap=0)
    try:
        documents = loader.load_and_split(text_splitter=text_splitter)
        for doc in documents:
            print(doc.page_content)
    except Exception as e:
        st.error(f"Failed to load the document: {e}")

    # Create/update Chroma vector store
    db = Chroma.from_documents(
        documents,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="okr_documents"
    )
    db.persist()
    st.sidebar.success("Document processed and indexed!")
else:
    # Try to load an existing index
    try:
        db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_name="okr_documents"
        )
        st.sidebar.info("Loaded existing document index.")
    except Exception as e:
        st.sidebar.warning(f"No document index found. Error: {e}")
        db = None

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
if db is not None:
    retriever = db.as_retriever(search_type="mmr", k=4)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, chain_type="stuff"
    )
else:
    qa_chain = None

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
