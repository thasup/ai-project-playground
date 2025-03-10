# OpenAI Chat Application

A simple Python application that uses OpenAI's GPT-3.5-turbo model to answer questions through the LangChain framework.

## Features

- Uses OpenAI's GPT-3.5-turbo model
- Implements LangChain for easy integration
- Environment variable support for secure API key management

## Prerequisites

- Python 3.11 or higher
- OpenAI API key
- pipenv (for dependency management)

## Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd pycode
```

2. Install dependencies using pipenv:
```bash
pipenv install
```

3. Create a `.env` file in the project root and add your OpenAI API key:
```bash
OPENAI_API_KEY=your-api-key-here
```

## Usage

1. Activate the virtual environment:
```bash
pipenv shell
```

2. Run the application:
```bash
python main.py
```

The application will ask a question about Thailand's capital and display the response.

## Project Structure

- `main.py`: Main application file containing the chat implementation
- `Pipfile`: Dependency management file
- `.env`: Environment variables file (not included in repository)

## Dependencies

- langchain-openai
- openai
- python-dotenv