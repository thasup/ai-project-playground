import json
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder, SystemMessage

# Create a prompt template that includes conversation history.
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant that maintains context of the conversation."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{content_json}")
])

# Example conversation history (list of message dicts)
chat_history = [
    {"role": "assistant", "content": "Hello, how can I help you today?"}
]

# Multi-modal human input as a list of dictionaries
human_input = [
    {"type": "text", "text": "Describe the weather in this image"},
    {"type": "image_url", "image_url": "data:image/jpeg;base64,<base64_encoded_data>"}
]

# Serialize the multi-modal input to JSON
human_input_json = json.dumps(human_input)

# Format the messages with both chat_history and the JSON-serialized human input.
formatted_messages = prompt.format_messages(content_json=human_input_json, chat_history=chat_history)

print(formatted_messages)
