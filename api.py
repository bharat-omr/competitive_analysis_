# market_research_agent.py

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.tools import Tool
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import time

# Load environment variables
load_dotenv()

# Wrap SerpAPI as a LangChain Tool manually
serp_tool = Tool(
    name="serp_search",
    func=SerpAPIWrapper().run,
    description="Use this tool to perform web searches via SerpAPI."
)

# Initialize the Tavily search tool
search_tool_tavily = TavilySearchResults(max_results=5)

# Agent state definition
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

# Agent class for business planning and market research
class MarketResearchAgent:
    def __init__(self, model, tools, system_prompt=""):
        self.system = system_prompt
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_model)
        graph.add_node("action", self.execute_tools)
        graph.add_conditional_edges("llm", self.needs_tool, {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def needs_tool(self, state: AgentState):
        return len(state['messages'][-1].tool_calls) > 0

    def call_model(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        response = self.model.invoke(messages)
        return {'messages': [response]}

    def execute_tools(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for call in tool_calls:
            print(f"Calling: {call}")
            if call['name'] not in self.tools:
                result = "Tool not found, please retry."
            else:
                result = self.tools[call['name']].invoke(call['args'])
            results.append(ToolMessage(tool_call_id=call['id'], name=call['name'], content=str(result)))
        return {'messages': results}

# System prompt for business planning
system_prompt = """
Hello! I’m your AI market analysis assistant. I’ll help you analyze the market for your business.
Ask follow-up questions in a helpful, strategic tone. Use research tools only when needed.
For example:
"Great choice! The sustainable fashion market is growing rapidly. Let’s start by defining your target audience and unique value proposition. Would you like me to conduct market research on eco-friendly clothing trends?"

Only when you have enough information, say: 'Here is your market analysis summary.' and provide a clear market insight using online tools.
"""

# Initialize model and agent
llm_model = ChatOpenAI(model="gpt-4")
agent = MarketResearchAgent(llm_model, [search_tool_tavily, serp_tool], system_prompt=system_prompt)

# FastAPI setup
app = FastAPI()

class UserMessage(BaseModel):
    conversation: list[str]

@app.post("/analyze")
async def analyze_market(user_input: UserMessage):
    conversation = [HumanMessage(content=msg) for msg in user_input.conversation]
    result = agent.graph.invoke({"messages": conversation})
    last_response = result['messages'][-1].content
    return {"response": last_response}

# CLI fallback to run locally for testing
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        uvicorn.run("market_research_agent:app", host="0.0.0.0", port=8000, reload=True)
    else:
        print("\n🤖: Hello! I’m your AI business planning assistant.")
        print("     I’ll ask a few questions to help build your market research.")
        user_input = input("👤: ")
        conversation = [HumanMessage(content=user_input)]

        while True:
            result = agent.graph.invoke({"messages": conversation})
            message = result['messages'][-1]

            print("\n🤖:", message.content)
            if "market analysis summary" in message.content.lower():
                break

            user_input = input("👤: ")
            conversation.append(HumanMessage(content=user_input))
            conversation.extend(result['messages'])