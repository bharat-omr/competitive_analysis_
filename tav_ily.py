from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")  # Tavily key from .env

# Initialize Gemini LLM
llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

# Competitive Intelligence Search Tool
tavily_search_tool = TavilySearch(
    max_results=5,
    topic="news"
)

# Create ReAct-style agent using LangGraph
agent = create_react_agent(llm, [tavily_search_tool])

# Ask Competitive Analysis Query
user_input = """
Give a competitive analysis of the Indian footwear market. 
Include insights about top players, their pricing, marketing strategies, target audiences, 
and opportunities for a new budget-friendly shoe brand. Cite recent sources.
"""

# Run agent and get only final output
final_output = None
for step in agent.stream({"messages": user_input}, stream_mode="values"):
    final_output = step["messages"][-1].content

# Print clean result
print("\n📊 Competitive Analysis Report\n")
print(final_output)
