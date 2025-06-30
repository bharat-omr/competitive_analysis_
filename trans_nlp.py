import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI
import google.generativeai as genai

# Load API keys
load_dotenv()
os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize tools
search_serp = SerpAPIWrapper()
search_tavily = TavilySearchResults(max_results=4)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# LLM required for memory summarization
llm_memory_model = ChatOpenAI(model_name="gpt-4")

# Initialize memory buffer with LLM
memory = ConversationSummaryBufferMemory(llm=llm_memory_model, max_token_limit=500)

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
                    # Add user input to memory
                    memory.chat_memory.add_user_message(user_input)

                    # Create memory context from history summary
                    memory_summary = memory.buffer if memory.buffer else "No prior context."

                    # Web search from both tools
                    serp_result = search_serp.run(user_input)
                    tavily_result = search_tavily.run(user_input)

                    # Prompt to Gemini
                    prompt = f"""
Hello! I’m your AI market analysis assistant. I’ll help you analysis market for your business.
Ask follow-up questions in a helpful, strategic tone. Use research tools only when needed.
For example:
"Great choice! The sustainable fashion market is growing rapidly. Let’s start by defining your target audience and unique value proposition. Would you like me to conduct market research on eco-friendly clothing trends?"

Only when you have enough information, say: 'Here is your market analysis summary.' and provide a clear market insight using online tools.

**SerpAPI:**
{serp_result}

**Tavily:**
{tavily_result}

🧠 Memory Summary:
{memory_summary}

💬 Current User Question:
User: {user_input}

Now respond as BizAI in a friendly, structured, and helpful tone. Use bullet points and business-style formatting.
""".strip()

                    # Gemini response
                    gemini_response = gemini_model.generate_content(prompt)
                    response_text = gemini_response.text.strip()

                    # Add assistant response to memory
                    memory.chat_memory.add_ai_message(response_text)

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
        memory.clear()

# Helper function to remove prefix
def strip_bot(bot_response):
    return bot_response.replace("🤖 **BizAI:**", "").replace(
        "📌 Let me know if you'd like help drafting a section, finding suppliers, or exploring your competition.", ""
    ).strip()

if __name__ == "__main__":
    main()