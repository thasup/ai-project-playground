# 🎓 AI Project Playground

Welcome to the **AI Project Playground**, a hands-on web application designed to help me learn and experiment with various AI concepts.

This repository provides a practical environment for building small AI projects that teach essential concepts such as Large Language Models (LLMs), chains, memory management, vector databases, Retrieval-Augmented Generation (RAG), and AI agents.

## 🚀 Overview

This playground is built using Streamlit and OpenAI's GPT models, offering an interactive interface for users to explore and understand the underlying principles of AI development. Whether you're a beginner or an experienced developer, this project will provide you through the material example of creating AI applications.

## 🎥 Demo

**OKRs Generator**:

**Technique**: AI-powered goal setting and alignment checking using multiple LLMs chaining with Langchain.

**Demo**: [Link](https://thasup-okr-generator.streamlit.app/)

https://github.com/user-attachments/assets/95b7bd3e-688e-4078-b9e5-cf37429da8f8

**AI Chat**:

**Technique**: Interactive conversation management with memory techniques (Summary and Buffer).

**Demo**: [Link](https://thasup-chat-bot.streamlit.app/)

https://github.com/user-attachments/assets/c0e6cd68-8bbb-4975-8b32-e59df6de12b8

## ✨ Features

### Learning Modules
- **OKRs Generator**:
  - 🤖 Generate Objectives and Key Results using AI.
  - 🎯 Understand how to structure goals and measurable outcomes.
  - 💡 Explore the alignment of initiatives with company values.

- **AI Chat**:
  - 💬 Engage in interactive conversations with memory capabilities.
  - 🧠 Experiment with different memory techniques (Summary vs. Buffer).
  - 🔄 Maintain context across conversations.

### Hands-On Experience
- **LLM Exploration**: Learn how to implement and interact with various OpenAI models.
- **Chain Management**: Understand how to create and manage chains of operations in AI workflows.
- **Memory Techniques**: Discover how to implement memory in AI applications for context retention.
- **Vector Databases**: Explore how to use vector databases for efficient data retrieval.
- **RAG Implementation**: Learn how to enhance AI responses with external data sources.
- **AI Agents**: Experiment with building intelligent agents that can perform tasks autonomously.

## 🛠️ Quick Start

1. **Clone the repository**:
   ```bash
   git clone <your-repository-url>
   cd ai-project-playground
   ```

2. **Install dependencies using pipenv**:
   - **Method 1 - Use PIPENV**:
     ```bash
     pipenv install
     ```
   - **Method 2 - Use UV**:
     ```bash
     uv pip install -r pyproject.toml
     ```

3. **Create a `.env` file in the project root and add your API keys**:
   ```bash
   OPENAI_API_KEY=your-openai-api-key-here
   GOOGLE_API_KEY=your-google-api-key-here
   ```

4. **Run the application** (choose one method):
   - **Method 1 - Use PIPENV to activate virtual environment first**:
     ```bash
     pipenv shell
     streamlit run apps/okr-app.py
     ```
   - OR Run directly with pipenv
     ```bash
     pipenv run streamlit run apps/okr-app.py
     ```
   - **Method 2 - Use UV to activate virtual environment first**:
     ```bash
     uv venv
     streamlit run apps/okr-app.py
     ```
   - OR Run directly with UV
     ```bash
     uv run streamlit run apps/okr-app.py
     ```

## 🛠️ Configuration

The application can be configured through the web interface:

### For OKR Generator
- **Model Selection**: Choose from available OpenAI models:
  - GPT-3.5 Turbo (fastest, most cost-effective)
  - GPT-4 Turbo Preview (latest model)
  - GPT-4 (most capable)
  - GPT-4 Mini (balanced performance)

### For AI Chat
- **Memory Technique**: Choose between:
  - **Summary Memory**: Summarizes conversation history for efficiency.
  - **Buffer Memory**: Stores full conversation history.

- **API Key**: Securely input your OpenAI and Google API keys in the sidebar.

## 📝 Usage

### For OKR Generator
1. Enter your strategic priorities in the left text area.
2. Input your company values in the right text area.
3. Click "Generate OKRs".
4. View results in organized sections:
   - Objective
   - Key Results
   - Initiatives
   - Alignment Check

### For AI Chat
1. Start a conversation by typing in the chat input.
2. Upload images if needed.
3. Choose the memory technique to maintain context.
4. Interact with the AI and view responses in real-time.

## 🏗️ Project Structure

```
ai-project-playground/
├── app.py              # Main Streamlit application for OKR generation
├── chat-app.py         # Streamlit application for interactive AI chat
├── Pipfile             # Dependency management
├── .env                # Environment variables (not in repo)
└── README.md           # This file
```

## 📦 Dependencies

- langchain-openai
- langchain-google-genai
- openai
- python-dotenv
- streamlit
- watchdog (for improved performance)

## 🔒 Security

- API keys are stored securely in environment variables.
- No API keys are stored in the code or repository.
- Secure password input field for API key entry.

---

This playground is designed to be a learning resource for anyone interested in AI development. Dive in, experiment, and enhance your understanding of AI concepts!
