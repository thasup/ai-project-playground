import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os
import time
import base64
import json

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="AI Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for memory and chat history
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
                ["gpt-4o-mini", "gpt-o1-mini", "gpt-3.5-turbo", "gpt-4-turbo-preview", "gpt-4-vision-preview"],
                help="Choose the OpenAI model for chat"
            )
        else:
            model = st.selectbox(
                "Select Google Model",
                ["gemini-2.0-flash", "gemini-1.5-flash"],
                help="Choose the Google model for chat"
            )
    with st.expander("Memory Settings", expanded=True):
        if st.button("Clear Conversation History"):
            st.session_state.memory.clear()
            st.session_state.messages = [{"role": "assistant", "content": "Conversation history cleared! 👋"}]
            st.success("Conversation history cleared!")
    # New expander for system prompt
    with st.expander("AI Assistant Settings", expanded=True):
        if "system_prompt" not in st.session_state:
            st.session_state.system_prompt = "You are a helpful AI assistant that maintains context of the conversation."
        user_prompt = st.text_area("System Prompt", value=st.session_state.system_prompt)

        # Confirm button for system prompt
        if st.button("Confirm System Prompt"):
            st.session_state.system_prompt = user_prompt  # Update session state with user-defined prompt
            st.success("System prompt updated!")

# Set API keys as environment variables
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["GOOGLE_API_KEY"] = google_api_key

# Initialize the LLM based on the chosen provider and provided API key
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

# Create a prompt template that includes conversation history
prompt = ChatPromptTemplate.from_messages([
    ("system", st.session_state.system_prompt),  # Use the user-defined prompt
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{content}"),
])

# Create the chain with memory and output parser
chain = prompt | llm | StrOutputParser()

# Main chat interface
st.title("💬 AI Chat")
st.markdown("Chat with an AI that remembers your conversation context")

# Display chat history
for message in st.session_state.messages:
    print("message >>>", message)
    with st.chat_message(message["role"]):
        if message.get("content"):
            st.markdown(message["content"])
        if message.get("files"):
            # Iterate over stored file dictionaries (base64 data is saved instead of file names)
            for file_dict in message["files"]:
                st.markdown(f"![{file_dict['name']}](data:{file_dict['mime']};base64,{file_dict['data']})")

# Accept user input with text and file upload support
user_input = st.chat_input(
    "What is up?",
    accept_file=True,
    file_type=["jpg", "jpeg", "png", "gif", "bmp"],
)
print("user_input >>>", user_input)

if user_input:
    # Process text input
    user_text = user_input["text"]
    print("user_text >>>", user_text)

    # Process uploaded files: encode each image to base64 and store info in a list
    uploaded_images = []
    if user_input.get("files"):
        for file in user_input["files"]:
            encoded_image = base64.b64encode(file.getbuffer()).decode()
            uploaded_images.append({
                "name": file.name,
                "data": encoded_image,
                "mime": file.type
            })

    # Save the user message along with image data in session state
    st.session_state.messages.append({
        "role": "user",
        "content": user_text,
        "files": uploaded_images
    })

    # Display user message in chat interface
    with st.chat_message("user"):
        if user_text:
            st.markdown(user_text, unsafe_allow_html=True)
        if uploaded_images:
            for img in uploaded_images:
                st.image(f"data:{img['mime']};base64,{img['data']}", caption=img['name'])

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            with st.spinner("Thinking..."):
                chat_history = st.session_state.memory.load_memory_variables({})["chat_history"]

                # Save user message to memory (store text and file names)
                st.session_state.memory.save_context(
                    {"content": json.dumps([{"text": user_text, "files": [img["name"] for img in uploaded_images]}])},
                    {"output": ""}
                )
                print("After saving context >>>", st.session_state.memory.load_memory_variables({}))

                # Prepare content list for the AI
                content_list = [{"type": "text", "text": user_text}]
                for img in uploaded_images:
                    content_list.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{img['mime']};base64,{img['data']}"
                        },
                    })

                human_message = HumanMessage(content=content_list)

                # Get response from the AI chain
                response = chain.invoke({
                    "content": content_list,
                    "chat_history": chat_history
                })
                print("response >>>", response)

                # Simulate typing by gradually revealing the response
                for chunk in response.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response, unsafe_allow_html=True)

                # Save the AI response to memory
                st.session_state.memory.save_context(
                    {"content": "",},
                    {"output": full_response}
                )

                # Append the assistant's response to the chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                })

        except Exception as e:
            print("Error >>>", e)
            message_placeholder.markdown(f"Error: {str(e)}")
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}"})

    # Rerun to update the interface with new messages
    st.rerun()