from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--language", type=str, default="Python")
parser.add_argument("--task", type=str, default="print 'Hello, world!'")
args = parser.parse_args()

load_dotenv()

# Create a prompt template
code_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that provides accurate and concise answers."),
    ("human", "Write a short {language} function that will {task}")
])

# Create the chat model
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Create a chain that combines the prompt and the model
chain = code_prompt | llm

# Use the chain with a question
result = chain.invoke({"language": args.language, "task": args.task})
print(result.content)