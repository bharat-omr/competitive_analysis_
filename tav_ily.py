import os
import datetime
from dotenv import load_dotenv
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage
from langchain_tavily import TavilySearch, TavilyCrawl

# Load environment variables
load_dotenv()
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# Init LLM
llm = init_chat_model(model="gpt-4.1-mini", model_provider="openai", temperature=0)

# Tools
tavily_search = TavilySearch(max_results=5)
tavily_crawl = TavilyCrawl()

# Prompt with Dynamic Task Instructions
today = datetime.datetime.today().strftime("%Y-%m-%d")
prompt = ChatPromptTemplate.from_messages([
    ("system", f"""You are an intelligent, multi-domain AI research assistant. Today is {today}.

You must:
- Understand user queries across domains (e.g., business, education, tech, health).
- Use real-time tools like web search and web crawl to find data.
- Format answers according to task type:
    - For research reports or business queries: include Executive Summary, Market Analysis, etc.
    - For factual Q&A: answer clearly with citations.
    - If user input is vague, ask clarifying follow-up.
    - Avoid hallucinating; prefer citing actual sources from search results.
"""),
    MessagesPlaceholder(variable_name="messages"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create agent + executor
agent = create_openai_tools_agent(
    llm=llm,
    tools=[tavily_search, tavily_crawl],
    prompt=prompt
)
agent_executor = AgentExecutor(agent=agent, tools=[tavily_search, tavily_crawl], verbose=True)

# Example user prompt (can change for any use-case)
user_input = "I want a strategy for launching an online AI learning platform for Indian college students. Include market scope, pricing, and outreach plan."

# Run agent
response = agent_executor.invoke({
    "messages": [HumanMessage(content=user_input)]
})

# Output
print("\n🧠 AI Assistant Response:\n")
print(response.get("output") or response)
