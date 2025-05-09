from langchain_openai import ChatOpenAI


from langchain_core.messages import HumanMessage

from langchain_core.messages import AIMessage

from langchain_core.messages import SystemMessage

import streamlit as st
from streamlit_chat import message

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)


from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain.globals import set_verbose

set_verbose(True)

history = FileChatMessageHistory('.chat_history.json')

memory: ConversationBufferMemory = ConversationBufferMemory(
    memory_key='chat_history',
    chat_memory=history,
    return_messages=True
)

st.set_page_config(
  page_title = 'Your Custom Assistant',
  page_icon = '👽'
  )
st.subheader('Your Custom ChatGPT')

chat = ChatOpenAI(model_name = 'gpt-3.5-turbo', temperature = 0.5)

if 'messages' not in st.session_state:
    st.session_state.messages = memory.chat_memory.messages

# Check if there's already a system message from memory
system_message = next((msg for msg in st.session_state.messages if isinstance(msg, SystemMessage)), None)

with st.sidebar:
    system_message_input = st.text_input(label = "System role", value=system_message.content if system_message else "")
    user_prompt = st.text_input(label = "Send a message")

    if not system_message and system_message_input:
        system_message = SystemMessage(content=system_message_input)
        st.session_state.messages.append(system_message)
        memory.chat_memory.add_message(system_message)

    if user_prompt:
        st.session_state.messages.append(
            HumanMessage(content = user_prompt)
        )
        memory.chat_memory.add_message(HumanMessage(content=user_prompt))

        with st.spinner('Working on your request'):
            response = chat.invoke(st.session_state.messages)
            st.session_state.messages.append(AIMessage(content = response.content))
            memory.chat_memory.add_message(AIMessage(content=response.content))

if len(st.session_state.messages)>0:
    for i,msg in enumerate(st.session_state.messages[1:]):
        if isinstance(msg, HumanMessage):
            message(msg.content, key=f'{i}+ HM', is_user=True)
        if isinstance(msg,AIMessage):
            message(msg.content, key=f'{i}+ AIM',is_user=False)