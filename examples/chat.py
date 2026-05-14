from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

prompt = ChatPromptTemplate(
  input_variables=["content"],
  messages=[
    ("system", "You are a helpful assistant that provides accurate and concise answers."),
    ("human", "{content}")
  ]
)

llm = ChatOpenAI(
  model="gpt-3.5-turbo",
  openai_api_key=os.getenv("OPEN_ROUTER_API_KEY"),
  base_url="https://openrouter.ai/api/v1"
)

chain = prompt | llm

while True:
  content = input("You: ")
  response = chain.invoke({"content": content})
  print("Assistant:", response.content)