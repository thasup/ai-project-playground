from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import ChatOpenAI
import os

load_dotenv()

embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPEN_ROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")

db = Chroma(
  persist_directory="emb",
  embedding_function=embeddings,
  collection_name="andreas_workshop"
)

retriever = db.as_retriever()

openai_llm = ChatOpenAI(
  model="gpt-3.5-turbo",
  openai_api_key=os.getenv("OPEN_ROUTER_API_KEY"),
  base_url="https://openrouter.ai/api/v1"
)

qa = RetrievalQA.from_chain_type(
  llm=openai_llm,
  retriever=retriever,
  chain_type="stuff"
)

result = qa.run("How can I prioritize my OKRs?")

print(result)