import os, tempfile
import pinecone
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma, Pinecone
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import streamlit as st

load_dotenv()

TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('data', 'vector_store')

st.set_page_config(
    page_title="Legal Document Expert",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================
# Main Chat Interface
# ========================
st.title("⚖️ Legal Document Expert")
st.markdown(
    "Use this tool to plan, create, execute, track, and review your legal documents. "
)
st.markdown(
    "Ask questions like *'How should I prioritize my legal documents?'* or share your progress for feedback."
)

def load_documents():
    loader = DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.pdf')
    documents = loader.load()
    return documents

def split_documents(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    return texts

def embeddings_on_local_vectordb(texts):
    vectordb = Chroma.from_documents(
        texts, embedding=OpenAIEmbeddings(openai_api_key=st.session_state.openai_api_key),
        persist_directory=LOCAL_VECTOR_STORE_DIR.as_posix()
    )
    vectordb.persist()
    retriever = vectordb.as_retriever(search_kwargs={'k': 5})
    return retriever

def embeddings_on_pinecone(texts):
    pinecone.init(api_key=st.session_state.pinecone_api_key, environment=st.session_state.pinecone_env)
    embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.openai_api_key)
    vectordb = Pinecone.from_documents(texts, embeddings, index_name=st.session_state.pinecone_index)
    retriever = vectordb.as_retriever()
    return retriever

def query_llm(retriever, query):
    # qa_chain = ConversationalRetrievalChain.from_llm(
    #     llm=OpenAIChat(openai_api_key=st.session_state.openai_api_key),
    #     retriever=retriever,
    #     return_source_documents=True,
    # )
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4o-mini", openai_api_key=st.session_state.openai_api_key),
        retriever=retriever,
        chain_type="stuff"
    )

    # Ensure the input dictionary has the correct keys
    result = qa_chain({'query': query, 'chat_history': st.session_state.messages})

    # Extract the answer from the result
    answer = result['result']  # Adjusted to match the expected output structure
    st.session_state.messages.append((query, answer))
    return answer

def input_fields():
    #
    with st.sidebar:
        #
        if "openai_api_key" in st.secrets:
            st.session_state.openai_api_key = st.secrets.openai_api_key
        else:
            st.session_state.openai_api_key = st.text_input("OpenAI API key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        #
        if "pinecone_api_key" in st.secrets:
            st.session_state.pinecone_api_key = st.secrets.pinecone_api_key
        else:
            st.session_state.pinecone_api_key = st.text_input("Pinecone API key", type="password", value=os.getenv("PINECONE_API_KEY", ""))
        #
        if "pinecone_env" in st.secrets:
            st.session_state.pinecone_env = st.secrets.pinecone_env
        else:
            st.session_state.pinecone_env = st.text_input("Pinecone environment", value=os.getenv("PINECONE_ENV", ""))
        #
        if "pinecone_index" in st.secrets:
            st.session_state.pinecone_index = st.secrets.pinecone_index
        else:
            st.session_state.pinecone_index = st.text_input("Pinecone index name", value=os.getenv("PINECONE_INDEX", ""))
    #
    st.session_state.pinecone_db = st.toggle('Use Pinecone Vector DB')
    #
    st.session_state.source_docs = st.file_uploader(label="Upload Documents", type="pdf", accept_multiple_files=True)
    #
    # Set API keys as environment variables
    os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key
    os.environ["PINECONE_API_KEY"] = st.session_state.pinecone_api_key
    os.environ["PINECONE_ENV"] = st.session_state.pinecone_env
    os.environ["PINECONE_INDEX"] = st.session_state.pinecone_index

def process_documents():
    if not st.session_state.openai_api_key or not st.session_state.pinecone_api_key or not st.session_state.pinecone_env or not st.session_state.pinecone_index or not st.session_state.source_docs:
        st.warning(f"Please upload the documents and provide the missing fields.")
    else:
        try:
            # Ensure the temporary directory exists
            TMP_DIR.mkdir(parents=True, exist_ok=True)

            for source_doc in st.session_state.source_docs:
                #
                with tempfile.NamedTemporaryFile(delete=False, dir=TMP_DIR.as_posix(), suffix='.pdf') as tmp_file:
                    tmp_file.write(source_doc.read())
                #
                documents = load_documents()
                #
                for _file in TMP_DIR.iterdir():
                    temp_file = TMP_DIR.joinpath(_file)
                    temp_file.unlink()
                #
                texts = split_documents(documents)
                #
                if not st.session_state.pinecone_db:
                    st.session_state.retriever = embeddings_on_local_vectordb(texts)
                else:
                    st.session_state.retriever = embeddings_on_pinecone(texts)
        except Exception as e:
            st.error(f"An error occurred: {e}")

def boot():
    #
    input_fields()
    #
    st.button("Submit Documents", on_click=process_documents)
    #
    if "messages" not in st.session_state:
        st.session_state.messages = []
    #
    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('ai').write(message[1])
    #
    if query := st.chat_input():
        st.chat_message("human").write(query)
        response = query_llm(st.session_state.retriever, query)
        st.chat_message("ai").write(response)

if __name__ == '__main__':
    #
    boot()
