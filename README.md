# 🎯 OKR Generator & AI Chat

A powerful web application that helps organizations generate effective Objectives and Key Results (OKRs) using AI, as well as providing a chat interface for interactive conversations. Built with Streamlit and OpenAI's GPT models.

## 🎥 Demo

https://github.com/user-attachments/assets/95b7bd3e-688e-4078-b9e5-cf37429da8f8

## ✨ Features

### OKR Generator
- 🤖 AI-powered OKR generation
- 🎯 Generate Objectives based on strategic priorities
- 📊 Create measurable Key Results
- 💡 Suggest relevant Initiatives
- 🔍 Alignment check with company values
- 🎨 Clean, intuitive web interface
- 🔄 Multiple OpenAI model support
- 🔒 Secure API key management

### AI Chat
- 💬 Interactive chat interface with memory capabilities
- 🧠 Experiment between Conversation Summary Memory and Conversation Buffer Memory
- 🔄 Maintain context of conversations
- 🎨 User-friendly design
- 🎹 Smooth Typing effect

## 🚀 Quick Start

1. Clone the repository:
   ```bash
   git clone <your-repository-url>
   cd okr-generator
   ```

2. Install dependencies using pipenv:
   ```bash
   pipenv install
   ```

3. Create a `.env` file in the project root and add your OpenAI and Google API keys:
   ```bash
   OPENAI_API_KEY=your-openai-api-key-here
   GOOGLE_API_KEY=your-google-api-key-here
   ```

4. Run the application (choose one method):

   **Method 1 - Activate virtual environment first (recommended for development):**
   ```bash
   pipenv shell
   streamlit run app.py
   ```

   **Method 2 - Run directly with pipenv:**
   ```bash
   pipenv run streamlit run app.py
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
okr-generator/
├── app.py              # Main Streamlit application for OKR generation
├── chat-app.py         # Streamlit application for interactive AI chat
├── Pipfile             # Dependency management
├── .env                # Environment variables (not in repo)
└── README.md           # This file
```

## 📦 Dependencies

- langchain-openai
- openai
- python-dotenv
- streamlit
- watchdog (for improved performance)

## 🔒 Security

- API keys are stored securely in environment variables.
- No API keys are stored in the code or repository.
- Secure password input field for API key entry.
