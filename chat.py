import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="AI Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        color: black;
    }
    .chat-message.user {
        background-color: #e9ecef;
    }
    .chat-message.assistant {
        background-color: #f0f2f6;
    }
    .chat-message .message-content {
        margin-top: 0.5rem;
    }
    .stButton > button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history"
    )

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    with st.expander("API Keys", expanded=True):
        openai_api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        google_api_key = st.text_input("Google API Key", type="password", value=os.getenv("GOOGLE_API_KEY", ""))

    with st.expander("Model Settings", expanded=True):
        model_provider = st.radio(
            "Select Model Provider",
            ["OpenAI", "Google"],
            help="Choose which AI model provider to use"
        )
        
        if model_provider == "OpenAI":
            model = st.selectbox(
                "Select OpenAI Model",
                ["gpt-3.5-turbo", "gpt-4-turbo-preview", "gpt-4o-mini", "gpt-o1-mini"],
                help="Choose the OpenAI model for chat"
            )
        else:
            model = st.selectbox(
                "Select Google Model",
                ["gemini-2.0-flash", "gemini-1.5-flash"],
                help="Choose the Google model for chat"
            )

    # Memory settings
    with st.expander("Memory Settings", expanded=True):
        if st.button("Clear Conversation History"):
            st.session_state.memory.clear()
            st.success("Conversation history cleared!")

# Set API keys
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["GOOGLE_API_KEY"] = google_api_key

# Main chat interface
st.title("💬 AI Chat")
st.markdown("Chat with an AI that remembers your conversation context")

# Display chat history
chat_history = st.session_state.memory.load_memory_variables({})["chat_history"]
for message in chat_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.container():
        st.markdown(f'<div class="chat-message {role}">'
                   f'<div><strong>{"You" if role == "user" else "AI"}</strong></div>'
                   f'<div class="message-content">{message.content}</div>'
                   '</div>', unsafe_allow_html=True)

# Chat input
if openai_api_key and (model_provider == "OpenAI" or google_api_key):
    user_input = st.text_area("Your message:", key="user_input", height=100)
    
    if st.button("Send", type="primary"):
        if user_input:
            # Create the appropriate LLM based on selection
            if model_provider == "OpenAI":
                llm = ChatOpenAI(
                    model=model,
                    openai_api_key=openai_api_key
                )
            else:
                llm = ChatGoogleGenerativeAI(
                    model=model,
                    google_api_key=google_api_key
                )

            # Create prompt template with memory
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful AI assistant that maintains context of the conversation."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}")
            ])

            # Create chain with memory
            chain = prompt | llm | StrOutputParser()

            with st.spinner("Thinking..."):
                # Save user message to memory
                st.session_state.memory.save_context(
                    {"input": user_input}, 
                    {"output": ""}
                )
                
                # Get response from AI
                response = chain.invoke({
                    "input": user_input,
                    "chat_history": chat_history
                })
                
                # Save AI response to memory
                st.session_state.memory.save_context(
                    {"input": ""}, 
                    {"output": response}
                )
                
                # Force refresh to show new messages
                st.rerun()

else:
    if model_provider == "OpenAI":
        st.error("🔑 Please enter your OpenAI API key in the sidebar to use this tool.")
    else:
        st.error("🔑 Please enter your Google API key in the sidebar to use this tool.")