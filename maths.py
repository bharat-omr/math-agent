import streamlit as st
from typing import Iterator
from agno.agent import Agent
from agno.agent import Agent, RunResponse
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.utils.pprint import pprint_run_response
from dotenv import load_dotenv
import os

load_dotenv()  # take environment variables from .env

api_key = os.getenv("GOOGLE_API_KEY")

# Prompt formatter
def format_math_prompt(user_question: str) -> str:
    return f"""
You are a helpful tutor. Answer the following problem in detail.

Question: {user_question}

If needed, search the web for formulas or data, but always explain your steps.
"""

# Initialize Agent
agent = Agent(
    model=Gemini(id="gemini-2.0-flash"),
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True,
)

# Streamlit UI
st.set_page_config(page_title="Math & Search Tutor Agent", layout="wide")
st.title("🧠 Math & Search AI Tutor")

with st.form("ask_form"):
    user_question = st.text_area("Enter your math or general question:", height=150)
    submitted = st.form_submit_button("Ask")

if submitted and user_question.strip():
    with st.spinner("Thinking..."):
        st.markdown("### 📘 AI Response")
        response: RunResponse=agent.run(format_math_prompt(user_question), stream=True)
        result = pprint_run_response(response, markdown=True)
        st.markdown(response)
