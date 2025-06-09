from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain_community.utilities import SerpAPIWrapper
import os
from dotenv import load_dotenv

# Load API keys
load_dotenv()
serp_api_key = os.getenv("SERP_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize SerpAPI
search = SerpAPIWrapper(serpapi_api_key=serp_api_key)

# Define Tool with Competitive Focus
tools = [
    Tool(
        name="Web Competitive Intelligence Search",
        func=search.run,
        description="Use this tool to gather competitive analysis, market trends, product strategies, and business insights from real-time data."
    )
]

# Load LLM (GPT-4)
llm = ChatOpenAI(
    temperature=0,
    openai_api_key=openai_api_key,
    model="gpt-4"
)

# Initialize Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

# User Input Dynamically (like a chatbot)
while True:
    user_query = input("\n📌 Ask a Competitive Analysis Question (or type 'exit' to quit):\n> ")
    if user_query.lower() in ["exit", "quit"]:
        print("👋 Exiting AI Competitive Analysis Assistant.")
        break

    # Agent Query Template: Structured Response
    structured_query = f"""
You are an AI business analyst. Using up-to-date web results and your business knowledge, answer the user's query in a structured competitive analysis format.

User Question: {user_query}

Structure your answer with:
1. Brief Overview
2. Key Competitors
3. Pricing/Product Strategy (if applicable)
4. Target Customer Segment
5. Marketing & Sales Approach
6. Trends or Innovations
7. Risks & Opportunities
8. Strategic Suggestions

Respond in clear, bullet-point or paragraph format.
"""

    # Run Agent
    response = agent.run(structured_query)

    # Output Response
    print("\n📘 Competitive Intelligence Report\n")
    print(response)
