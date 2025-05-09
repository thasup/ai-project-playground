from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--strategic_priorities", type=str, default="Grow revenue, Expand market share, Improve customer retention")
parser.add_argument("--company_values", type=str, default="Customer Partnership, Know Your Customer, Focus on Results, Build Relationships, Individual Freedom, Own Your Work, Grow Daily, Work in Public, Long-Term Stability, Be Bold, Stay Lean, Go Long")
args = parser.parse_args()

load_dotenv()

#1: Prompt Template for Objective Generation
objective_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are the OKR expert that helps draft effective Objectives. Objectives should be qualitative, inspiring, memorable, and relevant to what we want to achieve."),
    ("human", "Generate a potential Objective based on the following high-level strategic priorities: {strategic_priorities}. Consider our company values: {company_values}.")
])

#2: Prompt Template for Key Result Generation
key_result_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are the OKR expert that helps draft effective Key Results for a given Objective. Key Results should be quantitative, specific, measurable, and outcome-driven targets that measure progress toward the objective. Each Objective should have 3-5 KRs."),
    ("human", "For the Objective: '{objective}', generate 3-5 potential Key Results. Please include a mix of leading and lagging indicators if possible. Consider focusing on areas like customer value, financial health, people engagement, and operational efficiency. Each Key Result should suggest a starting value, a target value, and a potential measurement method.")
])

#3: Prompt Template for Initiative Brainstorming
initiative_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are the OKR expert that brainstorms potential Initiatives to achieve specific Key Results. Initiatives are actions or experiments taken to achieve key results. Consider the following categories based on their impact and effort: Quick Wins (High Impact/Low Effort), Big Bets (High Impact/High Effort), Incremental Improvements (Low Impact/Low Effort), and Thankless Tasks (Low Impact/High Effort)."),
    ("human", "For these Key Results: '{key_results}' (part of the Objective: '{objective}'), brainstorm 3-5 potential Initiatives that could help us achieve it. Briefly categorize each initiative based on impact and effort.")
])

#4: Prompt Template for Alignment Checks with Company Values
alignment_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that checks if a drafted Objective aligns with the company's core values. Our company values are: {company_values}. Objectives should be relevant to what we want to achieve and align with our overall direction."),
    ("human", "Assess the following Objective: '{objective}'. Does it strongly align with our company values? Explain your reasoning.")
])

# Create the OpenAI chat model
openai_llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Create Gemini chat model for alignment checks
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Create chains that combine the prompts and the model
objective_chain = objective_prompt | openai_llm | StrOutputParser()
key_result_chain = key_result_prompt | openai_llm | StrOutputParser()
initiative_chain = initiative_prompt | openai_llm | StrOutputParser()
alignment_chain = alignment_prompt | gemini_llm | StrOutputParser()

# First, get the Objective
objective_result = objective_chain.invoke({"strategic_priorities": args.strategic_priorities, "company_values": args.company_values})

# Then, use the Objective to get the Key Results
key_result_result = key_result_chain.invoke({"objective": objective_result})

# Then, use the Key Results to get the Initiatives
initiative_result = initiative_chain.invoke({
    "key_results": key_result_result,
    "objective": objective_result
})

# Then, use the Objective to get the Alignment Check
alignment_result = alignment_chain.invoke({
    "objective": objective_result,
    "company_values": args.company_values
})

print("Generated Objective:")
print(objective_result)
print("\nGenerated Key Results:")
print(key_result_result)
print("\nGenerated Initiatives:")
print(initiative_result)
print("\nAlignment Check:")
print(alignment_result)