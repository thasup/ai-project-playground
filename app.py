import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="OKRs Generator",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton > button {
        width: 100%;
        margin-top: 1rem;
    }
    .output-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    h1 {
        color: #0e1117;
        margin-bottom: 2rem;
    }
    .subtitle {
        color: #475569;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Main title
st.title("🎯 OKRs Generator")
st.markdown('<p class="subtitle">Generate effective Objectives and Key Results aligned with your strategic priorities</p>', unsafe_allow_html=True)

# Sidebar for API key
with st.sidebar:
    st.header("⚙️ Configuration")
    
    with st.expander("API Keys", expanded=True):
        openai_api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        google_api_key = st.text_input("Google API Key", type="password", value=os.getenv("GOOGLE_API_KEY", ""))

    with st.expander("Model Settings", expanded=True):
        # OpenAI model selection
        st.subheader("OpenAI Model")
        openai_model = st.selectbox(
            "Select Model",
            ["gpt-3.5-turbo", "gpt-4-turbo-preview", "gpt-4o-mini", "gpt-o1-mini"],
            help="Choose the OpenAI model to use for generation"
        )

        # Google model selection
        st.subheader("Google Model")
        google_model = st.selectbox(
            "Select Model",
            ["gemini-2.0-flash", "gemini-2.0-1.5-flash"],
            help="Choose the Google model to use for generation"
        )

    # Save API keys to environment variables
    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["GOOGLE_API_KEY"] = google_api_key

    if not openai_api_key or not google_api_key:
        st.warning("⚠️ Please enter your OpenAI and Google API keys to use this tool.")

# Main content
if openai_api_key and google_api_key:
    # Input fields in a clean layout
    st.subheader("💡 Input Parameters")
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            strategic_priorities = st.text_area(
                "Strategic Priorities",
                value="Grow revenue, Expand market share, Improve customer retention",
                help="Enter your organization's strategic priorities, separated by commas",
                height=100
            )
        with col2:
            company_values = st.text_area(
                "Company Values",
                value="Customer Partnership, Know Your Customer, Focus on Results, Build Relationships, Individual Freedom, Own Your Work, Grow Daily, Work in Public, Long-Term Stability, Be Bold, Stay Lean, Go Long",
                help="Enter your company values, separated by commas",
                height=100
            )

    if st.button("🚀 Generate OKRs", type="primary"):
        with st.spinner("🔄 Generating your OKRs..."):
            # Create the chat models
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

            # Display results in clean containers
            st.subheader("📊 Generated OKRs")
            
            with st.container():
                st.markdown('<div class="output-box">', unsafe_allow_html=True)
                st.markdown("#### 🎯 Objective")
                st.write(objective_result)
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown('<div class="output-box">', unsafe_allow_html=True)
                st.markdown("#### 📈 Key Results")
                st.write(key_result_result)
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown('<div class="output-box">', unsafe_allow_html=True)
                st.markdown("#### 🚀 Initiatives")
                st.write(initiative_result)
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown('<div class="output-box">', unsafe_allow_html=True)
                st.markdown("#### ✅ Alignment Check")
                st.write(alignment_result)
                st.markdown("</div>", unsafe_allow_html=True)

else:
    st.error("🔑 Please enter your OpenAI and Google API keys in the sidebar to use this tool.")