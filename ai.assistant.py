import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain_community.utilities import SerpAPIWrapper

# Load .env keys
load_dotenv()
serp_api_key = os.getenv("SERP_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Setup tools
search = SerpAPIWrapper(serpapi_api_key=serp_api_key)
tools = [
    Tool(
        name="Web Competitive Intelligence Search",
        func=search.run,
        description="Get up-to-date info on competitors, trends, or markets using SerpAPI."
    )
]

# Setup LLM
llm = ChatOpenAI(temperature=0.3, model="gpt-4", openai_api_key=openai_api_key)

# Initialize agent
agent = initialize_agent(tools=tools, llm=llm, agent="zero-shot-react-description", verbose=True)

# Get user company context
print("🚀 Welcome to the AI Market Intelligence Assistant\n")
your_company = input("🔹 Enter your company/startup name: ")
your_product = input("🔹 Describe your product/service in a sentence: ")
your_target = input("🔹 Who is your main customer segment? ")

# Chat loop
while True:
    user_query = input("\n📌 Ask a Competitive/Market Analysis Question (or type 'exit' to quit):\n> ")
    if user_query.lower() in ['exit', 'quit']:
        print("👋 Exiting AI Assistant. Good luck with your strategy!")
        break

    structured_prompt = f"""
You are an AI assistant for market and competitive intelligence.

Company Name: {your_company}
Product/Service: {your_product}
Target Segment: {your_target}

User Query: {user_query}

Analyze the question from the user's company's perspective and generate insights using current market data.

Structure your answer as:
1. Market Overview
2. Key Competitors
3. Product/Service Differentiation
4. Target Audience Insights
5. Competitive Pricing/Go-to-Market Strategy
6. Marketing & Sales Channels
7. Emerging Trends or Tech
8. Risks & Opportunities
9. Actionable Suggestions for {your_company}
"""

    result = agent.run(structured_prompt)
    print("\n📘 Market Intelligence Report\n")
    print(result)
