import os
import uuid
import asyncio
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

from agents import (
    Agent,
    Runner,
    WebSearchTool,
    function_tool,
    handoff,
    trace,
)

from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Set up page configuration
st.set_page_config(
    page_title="Market Analysis Assistant",
    page_icon="📈",
    layout="wide"
)

# Check API key
if not os.environ.get("OPENAI_API_KEY"):
    st.error("Please set your OPENAI_API_KEY environment variable")
    st.stop()

# --- Data Models ---
class ResearchPlan(BaseModel):
    topic: str
    search_queries: list[str]
    focus_areas: list[str]

class ResearchReport(BaseModel):
    title: str
    outline: list[str]
    report: str
    sources: list[str]
    word_count: int

# --- Tool: Save Important Fact ---
@function_tool
def save_important_fact(fact: str, source: str = None) -> str:
    if "collected_facts" not in st.session_state:
        st.session_state.collected_facts = []

    st.session_state.collected_facts.append({
        "fact": fact,
        "source": source or "Not specified",
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })
    return f"Fact saved: {fact}"

# --- Agents ---
research_agent = Agent(
    name="Research Agent",
    instructions="""
You are a market researcher. For each search query, summarize web results into concise insights (max 300 words).
Focus on market trends, competitors, growth forecasts, customer behavior, and pricing.
Avoid fluff. Only include factual and strategic insights.
    """,
    model="gpt-4o-mini",
    tools=[WebSearchTool(), save_important_fact],
)

editor_agent = Agent(
    name="Editor Agent",
    handoff_description="Writes detailed market analysis reports",
    instructions="""
You are a market analyst. You will write a detailed, strategic market research report.
Start with an outline. Then write the full report in markdown (at least 1000 words).
Include sections on trends, competitors, opportunities, and recommendations.
Use data provided from research agent.
    """,
    model="gpt-4o-mini",
    output_type=ResearchReport,
)

triage_agent = Agent(
    name="Triage Agent",
    instructions="""
You are the coordinator of this market research operation.
1. Take the business idea and details from the user input.
2. Create a research plan with:
   - topic
   - search_queries (3-5)
   - focus_areas (3-5)
3. Hand off to the Research Agent to collect insights
4. Then hand off to the Editor Agent to compile the report
    """,
    handoffs=[handoff(research_agent), handoff(editor_agent)],
    model="gpt-4o-mini",
    output_type=ResearchPlan,
)

# --- UI & Session ---
st.title("📊 AI Market Analysis Assistant")

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4().hex[:16])
if "collected_facts" not in st.session_state:
    st.session_state.collected_facts = []
if "research_done" not in st.session_state:
    st.session_state.research_done = False
if "report_result" not in st.session_state:
    st.session_state.report_result = None
if "business_info" not in st.session_state:
    st.session_state.business_info = {}

# --- Assistant Welcome and Form ---
with st.chat_message("assistant"):
    st.markdown("👋 Hello! I’m your AI market research assistant. Let’s start by learning about your business idea.")

with st.form("business_form"):
    business_name = st.text_input("What is the name or concept of your business?")
    business_model = st.text_input("What is your business model? (e.g., e-commerce, SaaS, service-based)")
    target_audience = st.text_input("Who is your target audience?")
    location = st.text_input("Where will your business operate or serve?")
    revenue_strategy = st.text_input("How do you plan to generate revenue?")
    submitted = st.form_submit_button("Submit and Analyze")

# --- Main Logic ---
async def run_research_conversation():
    st.session_state.collected_facts = []
    st.session_state.research_done = False
    st.session_state.report_result = None

    business_summary = f"""
Business Name or Concept: {st.session_state.business_info['business_name']}
Business Model: {st.session_state.business_info['business_model']}
Target Audience: {st.session_state.business_info['target_audience']}
Location: {st.session_state.business_info['location']}
Revenue Strategy: {st.session_state.business_info['revenue_strategy']}
    """

    with trace("Market Analysis", group_id=st.session_state.conversation_id):
        with st.chat_message("assistant"):
            st.markdown("📋 Creating research plan...")

        triage_result = await Runner.run(triage_agent, business_summary)

        if hasattr(triage_result.final_output, 'topic'):
            plan = triage_result.final_output
            with st.chat_message("assistant"):
                st.markdown(f"📌 **Topic**: {plan.topic}")
                st.json({"search_queries": plan.search_queries, "focus_areas": plan.focus_areas})

        previous_fact_count = 0
        for _ in range(10):
            current = len(st.session_state.collected_facts)
            if current > previous_fact_count:
                with st.chat_message("assistant"):
                    for fact in st.session_state.collected_facts[-(current - previous_fact_count):]:
                        st.info(f"**Fact**: {fact['fact']}\n\n**Source**: {fact['source']}")
                previous_fact_count = current
            await asyncio.sleep(1)

        with st.chat_message("assistant"):
            st.markdown("📝 Creating full market research report...")

        try:
            report_result = await Runner.run(editor_agent, triage_result.to_input_list())
            st.session_state.report_result = report_result.final_output

            with st.chat_message("assistant"):
                st.success("✅ Report Ready!")
                preview = report_result.final_output.report[:500] + "..."
                st.markdown(preview)
        except Exception as e:
            st.error(f"Error during report creation: {str(e)}")

        st.session_state.research_done = True

# --- Trigger Run ---
if submitted:
    st.session_state.business_info = {
        "business_name": business_name,
        "business_model": business_model,
        "target_audience": target_audience,
        "location": location,
        "revenue_strategy": revenue_strategy
    }
    with st.spinner("Analyzing market..."):
        try:
            asyncio.run(run_research_conversation())
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            st.session_state.report_result = f"# Error\n\nCould not generate report.\n\n{str(e)}"
            st.session_state.research_done = True

# --- Report Display ---
if st.session_state.research_done and st.session_state.report_result:
    with st.expander("📄 Full Report", expanded=True):
        report = st.session_state.report_result
        if hasattr(report, 'report'):
            st.markdown(report.report)
        else:
            st.markdown(str(report))

    if hasattr(report, 'sources') and report.sources:
        with st.expander("📚 Sources"):
            for i, src in enumerate(report.sources):
                st.markdown(f"{i+1}. {src}")

    st.download_button("⬇️ Download Report", report.report, file_name="market_report.md", mime="text/markdown")
