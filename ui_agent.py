import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain_community.utilities import SerpAPIWrapper

# Load environment variables
load_dotenv()
serp_api_key = os.getenv("SERP_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize SerpAPI
search = SerpAPIWrapper(serpapi_api_key=serp_api_key)

# Define tools
tools = [
    Tool(
        name="Web Competitive Intelligence Search",
        func=search.run,
        description="Use this tool to gather competitive analysis, market trends, product strategies, and business insights from real-time data."
    )
]

# Initialize LLM
llm = ChatOpenAI(
    temperature=0,
    openai_api_key=openai_api_key,
    model="gpt-4"
)

# Initialize agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

# Streamlit UI
st.set_page_config(page_title="AI Competitive Intelligence", page_icon="📊")
st.title("📊 AI Competitive Intelligence Assistant")
st.write("Ask business-related questions and get a structured competitive analysis report.")

# Input box
user_query = st.text_input("📌 Enter your competitive analysis question:", "")

if user_query:
    with st.spinner("Analyzing..."):
        structured_query =  f"""
You are a smart, friendly AI business assistant named "BizAI".

🎯 Your Goal:
Help the user build a strong business plan by first collecting all essential details. Once sufficient info is gathered, perform a detailed **market analysis** including real-world data such as **CAGR** and competitor insights.

📋 Step-by-Step Instructions:

1. 📥 **Gather Information**
   Ask structured questions to collect the following:
   - ✅ Business Name
   - ✅ Type of Business (Product or Service?)
   - ✅ Target Customers / Audience
   - ✅ Industry or Niche
   - ✅ Location / Market Area
   - ✅ Unique Selling Proposition (USP)
   - ✅ Known Competitors (if any)
   - ✅ Pricing or Monetization Strategy

   If anything is missing, ask follow-up questions clearly and simply.

2. 📊 **Perform Market Research & Analysis**
   Once all business details are available, generate a Market Research Report using this structure:
   - **1. Overview** – Brief about the industry and business model.
   - **2. Target Audience** – Age, gender, behaviors, region.
   - **3. Competitor Landscape** – Main players and comparison.
   - **4. Market Trends & Size**  
     - Use recent data from web.
     - Include **CAGR (Compound Annual Growth Rate)** if available.
     - Example: "The industry is projected to grow from $2B in 2023 to $3.5B in 2028, at a CAGR of 11.5%."
   - **5. Opportunities & Risks**
   - **6. Recommended Next Steps**

3. 📌 **Suggestions & Execution**
   - Recommend what to do next: "Want me to help write your executive summary?" or "Shall I find suppliers?"
   - If user says **Yes**, continue without re-confirming.
   - Always stay focused unless user clearly switches the topic.

🔎 Web Search:
{user_query} """

        # Run agent and display result
        try:
            response = agent.run(structured_query)
            st.subheader("📘 Competitive Intelligence Report")
            st.markdown(response)
        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")
