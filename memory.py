import os
import uuid
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_community.utilities import SerpAPIWrapper
import google.generativeai as genai

# Load keys
load_dotenv()
os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

search = SerpAPIWrapper()
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# Helper to clean bot response
def strip_bot(bot_response):
    return bot_response.replace("🤖 **BizAI:**", "").replace(
        "📌 Let me know if you'd like help drafting a section, finding suppliers, or exploring your competition.", ""
    ).strip()

# Helper to get or assign session ID
def get_user_session():
    if "user_session_id" not in st.session_state:
        st.session_state.user_session_id = str(uuid.uuid4())[:8]  # simple unique ID
    return st.session_state.user_session_id

# Setup vector memory per user
def get_vectorstore(session_id):
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
    persist_dir = os.path.join("bizai_sessions", session_id)
    os.makedirs(persist_dir, exist_ok=True)
    return Chroma(persist_directory=persist_dir, embedding_function=embedding_model)

def main():
    st.set_page_config(page_title="🌐 BizAI - Business Assistant", layout="wide")
    st.title("🤖 BizAI - Multi-User Business Assistant")

    # Get or create session
    user_name = st.text_input("👤 Enter your name or session ID:", value="User1")
    session_id = user_name.strip().lower().replace(" ", "_")
    st.caption(f"🔐 Session ID: `{session_id}` — Your data will be kept separate.")

    vectorstore = get_vectorstore(session_id)

    if f"chat_history_{session_id}" not in st.session_state:
        st.session_state[f"chat_history_{session_id}"] = []

    # Display session chat
    for pair in st.session_state[f"chat_history_{session_id}"]:
        with st.chat_message("user"):
            st.markdown(pair["user"])
        with st.chat_message("assistant"):
            st.markdown(pair["bot"])

    # Chat input
    user_input = st.chat_input("What business idea are you working on?")
    if user_input:
        st.chat_message("user").markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Thinking and researching..."):
                try:
                    # Long memory retrieval
                    docs = vectorstore.similarity_search(user_input, k=5)
                    long_memory = "\n---\n".join([doc.page_content for doc in docs])

                    # Web search
                    serp_result = search.run(user_input)

                    # Prompt
                    prompt = f"""
You are a smart, friendly AI business assistant named "BizAI".

🎯 Your Goal:
Help the user build a strong business plan by first collecting all essential details. Once sufficient info is gathered, perform a detailed **market analysis** including real-world data such as **CAGR** and competitor insights.

📋 Step-by-Step Instructions:
1. 📥 **Gather Info** (name, type, audience, USP, competitors...)
2. 📊 **Market Analysis** (overview, audience, CAGR, trends, risks)
3. 📌 **Suggestions** for next steps

🔎 Web Search:
{serp_result}

🧠 Long-Term Context for {user_name}:
{long_memory}

💬 User Question:
User: {user_input}
                    """.strip()

                    gemini_response = gemini_model.generate_content(prompt)
                    response_text = gemini_response.text.strip()

                    final_answer = (
                        f"🤖 **BizAI:**\n\n{response_text}\n\n"
                        "📌 Let me know if you'd like help drafting a section, finding suppliers, or exploring your competition."
                    )

                    st.markdown(final_answer)

                    # Save to memory
                    st.session_state[f"chat_history_{session_id}"].append({
                        "user": user_input,
                        "bot": final_answer
                    })

                    vectorstore.add_documents([
                        Document(page_content=f"User: {user_input}\nAssistant: {response_text}")
                    ])

                except Exception as e:
                    st.error(f"❌ Error: {e}")

    st.divider()
    if st.button("🗑️ Clear This Session's Chat", use_container_width=True):
        st.session_state[f"chat_history_{session_id}"] = []
        st.success("✅ Chat cleared for this session.")

if __name__ == "__main__":
    main()
