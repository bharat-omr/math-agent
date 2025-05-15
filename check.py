import streamlit as st
import google.generativeai as genai
from langchain_community.utilities import SerpAPIWrapper
import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# Set API Keys from environment or fallback (🔐 Don't hardcode in production)
os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize models
gemini_model = genai.GenerativeModel("gemini-2.0-flash")
search = SerpAPIWrapper()  # Default to serpapi

# Streamlit UI
st.set_page_config(page_title="Ask AI + Web Search", layout="centered")
st.title("🤖 Ask AI with Web Search Power")

# Input field
question = st.text_area("Ask your question:", placeholder="e.g. What is the current inflation rate in the US?")

# User clicks submit
if st.button("Get Answer"):
    if question.strip():
        with st.spinner("Searching and thinking..."):
            try:
                # Get real-time info from web
                serp_result = search.run(question)

                # Create a detailed prompt
                prompt = f"""You are a helpful AI. Based on the real-time web result and your own knowledge, answer the following question thoroughly or step by step solution.

Question: {question}
Web Search Result: {serp_result}

Answer:"""

                # Ask Gemini with context
                response = gemini_model.generate_content(prompt)

                # Output
                st.markdown("### 🌐 Web + AI Answer")
                st.markdown(response.text.strip())

            except Exception as e:
                st.error(f"❌ Error: {e}")
    else:
        st.warning("Please enter a question first.")
