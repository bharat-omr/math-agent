# 🤖 AI PDF + Web Chat Assistant

A Streamlit-based AI assistant that allows users to chat with uploaded PDFs and ask real-time questions using web search. It uses Google Gemini, LangChain, and DSPy for intelligent responses and feedback analysis.

## 🚀 Features

- 📄 Upload and chat with multiple PDF documents
- 🌐 Ask real-time questions with web fallback using SerpAPI
- 🤖 Powered by Google Gemini (via LangChain + DSPy)
- 📡 Web fallback if PDF has no relevant info
- 🔎 AI feedback on response clarity and helpfulness

## 🛠️ Tech Stack

- Streamlit
- LangChain
- Google Generative AI (Gemini)
- SerpAPI (for web search)
- DSPy (for response evaluation)
- FAISS (for local document retrieval)

## 📦 Setup

1. Clone this repo:
   ```bash
   git clone https://github.com/bharat-omr/ai-pdf-web-chat.git
   
## Create a .env file:

GOOGLE_API_KEY=your_google_api_key
SERPAPI_API_KEY=your_serpapi_key

## Run the app:

streamlit run main.py
