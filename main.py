from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--language", type=str, default="Python")
parser.add_argument("--task", type=str, default="print 'Hello, world!'")
args = parser.parse_args()

load_dotenv()

# Create the first prompt template for code generation
code_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that provides accurate and concise answers. Make sure to return the code only, no other text."),
    ("human", "Write a short {language} function that will {task}")
])

# Create the second prompt template that uses the output from the first prompt
explanation_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that explains code in simple terms."),
    ("human", "Explain this code in simple terms: {code}")
])

# Create the chat model
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Create chains that combine the prompts and the model
code_chain = code_prompt | llm | StrOutputParser()
explanation_chain = explanation_prompt | llm

# First, get the code
code_result = code_chain.invoke({"language": args.language, "task": args.task})

# Then, use the code to get the explanation
explanation_result = explanation_chain.invoke({"code": code_result})

print("Generated Code:")
print(code_result)
print("\nExplanation:")
print(explanation_result.content)