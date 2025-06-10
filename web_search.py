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
    st.set_page_config(page_title="🌐 Smart AI Assistant", layout="wide")
    st.title("🤖 Conversational AI Web Assistant")

    st.write("Ask anything! I’ll search the web, understand your needs, and remember past answers to help you better.")

    # Memory
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    # Display chat history
    for pair in st.session_state.conversation_history:
        with st.chat_message("user"):
            st.markdown(pair["user"])
        with st.chat_message("assistant"):
            st.markdown(pair["bot"])

    # New user input
    user_input = st.chat_input("What do you want to know?")
    if user_input:
        st.chat_message("user").markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Thinking and searching..."):
                try:
                    # Build conversation context
                    context = "\n".join(
                        [f"User: {pair['user']}\nAssistant: {pair['bot']}" for pair in st.session_state.conversation_history[-5:]]
                    )

                    serp_result = search.run(user_input)
                    prompt = f"""
You are a smart, friendly AI assistant that helps the user based on their ongoing conversation.
Remember their past questions and give context-aware answers.
If something is unclear, ask clarifying questions.

Here is the recent conversation:
{context}

Current question: {user_input}
Web Search Result: {serp_result}

Answer (step-by-step and helpful):
                    """.strip()

                    gemini_response = gemini_model.generate_content(prompt)
                    final_answer = f"📡 *Web Search Answer*\n\n{gemini_response.text.strip()}"

                    st.markdown(final_answer)
                    st.session_state.conversation_history.append({"user": user_input, "bot": final_answer})

                except Exception as e:
                    st.error(f"❌ Error: {e}")

    # Clear chat button
    st.divider()
    if st.button("🗑️ Clear Chat Memory", use_container_width=True):
        st.session_state.conversation_history = []

if __name__ == "__main__":
    main()
