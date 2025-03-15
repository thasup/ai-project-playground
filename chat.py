import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os
import time

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="AI Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history"
    )
    st.session_state.messages = [{"role": "assistant", "content": "Let's start chatting! 👇"}]

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
            st.session_state.messages = [{"role": "assistant", "content": "Conversation history cleared! 👋"}]
            st.success("Conversation history cleared!")

# Set API keys
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["GOOGLE_API_KEY"] = google_api_key

# Main chat interface
st.title("💬 AI Chat")
st.markdown("Chat with an AI that remembers your conversation context")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("content"):
            st.markdown(message["content"])
        if message.get("files"):
            for file in message["files"]:
                # Save the file to the 'uploads' directory
                upload_dir = "uploads"
                os.makedirs(upload_dir, exist_ok=True)
                file_path = os.path.join(upload_dir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                
                # Display image or provide a download link based on file type
                if file.type.startswith("image/"):
                    st.image(file)
                else:
                    st.download_button(
                        label=f"Download {file.name}",
                        data=file,
                        file_name=file.name,
                        mime=file.type
                    )

# Accept user input
user_input = st.chat_input(
    "What is up?",
    accept_file=True,
    file_type=["jpg", "jpeg", "png"],
)

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({
        "role": "user",
        "content": user_input["text"],
        "files": user_input.get("files", [])
    })
    with st.chat_message("user"):
        st.markdown(user_input["text"])
        if user_input["files"]:
            st.image(user_input["files"][0])

    # Initialize LLM based on provider
    if model_provider == "OpenAI" and openai_api_key:
        llm = ChatOpenAI(
            model=model,
            openai_api_key=openai_api_key
        )
    elif model_provider == "Google" and google_api_key:
        llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=google_api_key
        )
    else:
        st.error("🔑 Please enter the required API key in the sidebar.")
        st.stop()

    # Create prompt template with memory
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant that maintains context of the conversation."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    # Create chain with memory
    chain = prompt | llm | StrOutputParser()

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            with st.spinner("Thinking..."):
                # Save user message to memory
                st.session_state.memory.save_context(
                    {"input": user_input}, 
                    {"output": ""}
                )
                
                # Get response from AI
                response = chain.invoke({
                    "input": user_input,
                    "chat_history": st.session_state.memory.load_memory_variables({})["chat_history"]
                })
                
                # Simulate typing with cursor
                for chunk in response.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            
            # Save AI response to memory
            st.session_state.memory.save_context(
                {"input": user_input}, 
                {"output": response}
            )
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        except Exception as e:
            message_placeholder.markdown(f"Error: {str(e)}")
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}"})

    # Refresh to show new messages
    st.rerun()
