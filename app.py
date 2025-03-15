import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="OKR Generator",
    page_icon="🎯",
    layout="wide"
)

# Title and description
st.title("🎯 OKR Generator")
st.markdown("""
This tool helps you generate Objectives and Key Results (OKRs) based on your strategic priorities and company values.
""")

# Sidebar for API key
with st.sidebar:
    st.header("Configuration")
    openai_api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))

    # OpenAI model selection
    openai_model = st.selectbox(
        "Select OpenAI Model",
        ["gpt-3.5-turbo", "gpt-4-turbo-preview", "gpt-4o-mini", "gpt-o1-mini"],
        help="Choose the OpenAI model to use for generation"
    )

    google_api_key = st.text_input("Google API Key", type="password", value=os.getenv("GOOGLE_API_KEY", ""))

    # Google model selection
    google_model = st.selectbox(
        "Select Google Model",
        ["gemini-2.0-flash", "gemini-2.0-1.5-flash"],
        help="Choose the Google model to use for generation"
    )

    # Save API keys to environment variables
    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["GOOGLE_API_KEY"] = google_api_key

    if not openai_api_key or not google_api_key:
        st.warning("Please enter your OpenAI and Google API keys in the sidebar to use this tool.")

# Main content
if openai_api_key and google_api_key:
    # Input fields
    col1, col2 = st.columns(2)
    with col1:
        strategic_priorities = st.text_area(
            "Strategic Priorities",
            value="Grow revenue, Expand market share, Improve customer retention",
            help="Enter your high-level strategic priorities, separated by commas"
        )

    with col2:
        company_values = st.text_area(
            "Company Values",
            value="Customer Partnership, Know Your Customer, Focus on Results, Build Relationships, Individual Freedom, Own Your Work, Grow Daily, Work in Public, Long-Term Stability, Be Bold, Stay Lean, Go Long",
            help="Enter your company values, separated by commas"
        )

    # Generate button
    if st.button("Generate OKRs", type="primary"):
        with st.spinner("Generating your OKRs..."):
            # Create the chat model
            openai_llm = ChatOpenAI(
                model=openai_model,
                openai_api_key=openai_api_key
            )

            gemini_llm = ChatGoogleGenerativeAI(
                model=google_model,
                google_api_key=google_api_key
            )

            # Create prompt templates
            objective_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are the OKR expert that helps draft effective Objectives. Objectives should be concise, qualitative, inspiring, memorable, and relevant to what we want to achieve."),
                ("human", "Generate only one potential Objective based on the following high-level strategic priorities: {strategic_priorities}. Consider our company values: {company_values}.")
            ])

            key_result_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are the OKR expert that helps draft effective Key Results for a given Objective. Key Results should be quantitative, specific, measurable, and outcome-driven targets that measure progress toward the objective. Each Objective should have 3-5 KRs."),
                ("human", "For the Objective: '{objective}', generate 3-5 potential Key Results. Please include a mix of leading and lagging indicators if possible. Consider focusing on areas like customer value, financial health, people engagement, and operational efficiency. Each Key Result should suggest a starting value, a target value, and a potential measurement method.")
            ])

            initiative_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are the OKR expert that brainstorms potential Initiatives to achieve specific Key Results. Initiatives are actions or experiments taken to achieve key results. Consider the following categories based on their impact and effort: Quick Wins (High Impact/Low Effort), Big Bets (High Impact/High Effort), Incremental Improvements (Low Impact/Low Effort), and Thankless Tasks (Low Impact/High Effort)."),
                ("human", "For these Key Results: '{key_results}' (part of the Objective: '{objective}'), brainstorm 3-5 potential Initiatives that could help us achieve it. Briefly categorize each initiative based on impact and effort.")
            ])

            alignment_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant that checks if a drafted Objective aligns with the company's core values. Our company values are: {company_values}. Objectives should be relevant to what we want to achieve and align with our overall direction."),
                ("human", "Assess the following Objective: '{objective}'. Does it strongly align with our company values? Explain your reasoning.")
            ])

            # Create chains
            objective_chain = objective_prompt | openai_llm | StrOutputParser()
            key_result_chain = key_result_prompt | openai_llm | StrOutputParser()
            initiative_chain = initiative_prompt | openai_llm | StrOutputParser()
            alignment_chain = alignment_prompt | gemini_llm | StrOutputParser()

            # Generate results
            objective_result = objective_chain.invoke({
                "strategic_priorities": strategic_priorities,
                "company_values": company_values
            })

            key_result_result = key_result_chain.invoke({"objective": objective_result})
            initiative_result = initiative_chain.invoke({
                "key_results": key_result_result,
                "objective": objective_result
            })
            alignment_result = alignment_chain.invoke({
                "objective": objective_result,
                "company_values": company_values
            })

            # Display results in tabs
            tab1, tab2, tab3, tab4 = st.tabs(["Objective", "Key Results", "Initiatives", "Alignment Check"])

            with tab1:
                st.markdown("### Generated Objective")
                st.write(objective_result)

            with tab2:
                st.markdown("### Generated Key Results")
                st.write(key_result_result)

            with tab3:
                st.markdown("### Generated Initiatives")
                st.write(initiative_result)

            with tab4:
                st.markdown("### Alignment Check")
                st.write(alignment_result)

else:
    st.error("Please enter your OpenAI and Google API keys in the sidebar to use this tool.") 