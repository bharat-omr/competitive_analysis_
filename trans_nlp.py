import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.utilities import SerpAPIWrapper
import google.generativeai as genai
from transformers import pipeline

# Load API keys
load_dotenv()
os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize tools
search = SerpAPIWrapper()
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# Load intent classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define possible intents
candidate_labels = [
    "business idea",
    "market research",
    "find suppliers",
    "draft business plan",
    "general question",
    "small talk"
]

# Extract intent
def detect_intent(text):
    result = classifier(text, candidate_labels)
    return result["labels"][0]  # top predicted label

# Intent handlers
def handle_market_research(text):
    result = search.run(text)
    prompt = f"Summarize these market trends clearly: {result}"
    return gemini_model.generate_content(prompt).text.strip()

def handle_find_suppliers(text):
    result = search.run(f"Top suppliers for {text}")
    prompt = f"List top suppliers from this: {result}"
    return gemini_model.generate_content(prompt).text.strip()

def handle_draft_plan(text):
    prompt = f"Draft the business plan section based on: {text}"
    return gemini_model.generate_content(prompt).text.strip()

def handle_general_question(text):
    return gemini_model.generate_content(text).text.strip()

def handle_small_talk(text):
    return "🙂 I'm here to help with anything. Tell me more!"

# Streamlit app
def main():
    st.set_page_config(page_title="🤖 Modular Smart Assistant", layout="wide")
    st.title("🧠 Smart Transformer-based AI Assistant")

    st.write("Ask about business, research, planning, or anything else!")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(chat["user"])
        with st.chat_message("assistant"):
            st.markdown(chat["bot"])

    user_input = st.chat_input("What's your question?")
    if user_input:
        st.chat_message("user").markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing and answering..."):
                try:
                    intent = detect_intent(user_input)

                    if intent == "market research":
                        answer = handle_market_research(user_input)
                    elif intent == "find suppliers":
                        answer = handle_find_suppliers(user_input)
                    elif intent == "draft business plan":
                        answer = handle_draft_plan(user_input)
                    elif intent == "business idea":
                        answer = f"Great idea! Do you want a market report or competitor analysis for '{user_input}'?"
                    elif intent == "general question":
                        answer = handle_general_question(user_input)
                    else:
                        answer = handle_small_talk(user_input)

                    st.markdown(answer)
                    st.session_state.chat_history.append({"user": user_input, "bot": answer})

                except Exception as e:
                    st.error(f"❌ Error: {e}")

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []

if __name__ == "__main__":
    main()
