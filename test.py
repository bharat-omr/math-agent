from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.agents import Tool, load_tools, create_react_agent, AgentExecutor
from langchain.tools import DuckDuckGoSearchResults

import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)

def create_search_math_agent():
    # Initialize Gemini LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

    # Improved ReAct prompt that emphasizes math step-by-step
    react_template =  """Answer the following question as best as you can. You have access to the following tools:

{tools}

Use the following format:

Question : The input question you must answer
Thought : you should always think about what to do
Action: The action to take, should be one of [{tool_names}]
Action Input : the input to the action
Observation : the result of the action
... (this thought/Action/Action input/Observation can repeat N times)
Thought: I know the final answer
Final Answer: The final answer to the original question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""


    prompt = PromptTemplate(
        template=react_template,
        input_variables=["tools", "tool_names", "input", "agent_scratchpad"]
    )

    # Define DuckDuckGo search tool
    search = DuckDuckGoSearchResults(backend="api")
    search_tool = Tool(
        name="DuckDuck search tool",
        description="A web search engine. Use this to search the internet for real-time data or general facts.",
        func=search.run,
    )

    # Prioritize math tool
    tools = load_tools(["llm-math"], llm=llm)
    tools.append(search_tool)

    # Create ReAct agent
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    return agent_executor

# 🔁 Run the agent with a sample math+search query
if __name__ == "__main__":
    agent_executor = create_search_math_agent()

    user_input = "ind vs pak war 2025 "
    result = agent_executor.invoke({"input": user_input})

    print("\n🧠 Final Answer:\n", result["output"])
