import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.utilities import SerpAPIWrapper
import google.generativeai as genai

# Load API keys
load_dotenv()
os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize tools
search = SerpAPIWrapper()
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# Streamlit Web App
def main():
    st.set_page_config(page_title="🌐 BizAI - Business Assistant", layout="wide")
    st.title("🤖 BizAI - Your Smart Business Planning Assistant")

    st.write("I’ll help you develop your business plan, research your market, and find real insights. Just tell me your idea!")

    # Session memory
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    # Display past messages
    for pair in st.session_state.conversation_history:
        with st.chat_message("user"):
            st.markdown(pair["user"])
        with st.chat_message("assistant"):
            st.markdown(pair["bot"])

    # New input
    user_input = st.chat_input("What business idea are you working on?")
    if user_input:
        st.chat_message("user").markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Thinking and researching..."):
                try:
                    # Create memory context from last 5 messages
                    recent_memory = "\n".join([
                        f"User: {pair['user']}\nAssistant: {strip_bot(pair['bot'])}"
                        for pair in st.session_state.conversation_history[-5:]
                    ])

                    # Web search
                    serp_result = search.run(user_input)

                    # Prompt to Gemini
                    prompt = f"""
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

🔎 Web Search Result (for current input):
{serp_result}

🧠 Memory Context (last 5 turns):
{recent_memory}

💬 Current User Question:
User: {user_input}

Now respond as BizAI in a friendly, structured, and helpful tone. Use bullet points and business-style formatting.
""".strip()


                    # Gemini response
                    gemini_response = gemini_model.generate_content(prompt)
                    response_text = gemini_response.text.strip()

                    final_answer = (
                        f"🤖 **BizAI:**\n\n{response_text}\n\n"
                        "📌 Let me know if you'd like help drafting a section, finding suppliers, or exploring your competition."
                    )

                    # Show response
                    st.markdown(final_answer)

                    # Save to conversation history
                    st.session_state.conversation_history.append({
                        "user": user_input,
                        "bot": final_answer
                    })

                except Exception as e:
                    st.error(f"❌ Error: {e}")

    # Clear history
    st.divider()
    if st.button("🗑️ Clear Chat Memory", use_container_width=True):
        st.session_state.conversation_history = []

# Helper function to remove prefix
def strip_bot(bot_response):
    return bot_response.replace("🤖 **BizAI:**", "").replace(
        "📌 Let me know if you'd like help drafting a section, finding suppliers, or exploring your competition.", ""
    ).strip()

if __name__ == "__main__":
    main()
