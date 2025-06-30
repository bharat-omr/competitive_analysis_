from fastapi import FastAPI
import requests
import google.generativeai as genai
import os
from dotenv import load_dotenv
import uvicorn

# Load Gemini API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

app = FastAPI()

def summarize_chunk(messages_chunk):
    chat_text = ""
    for msg in messages_chunk:
        role = "User" if msg.get("isUser") else "Assistant"
        chat_text += f"{role}: {msg['content']}\n"

    prompt = f"""
You are a helpful summarization assistant.

Summarize this short chat exchange in 1-2 sentences:
{chat_text}
"""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Error summarizing chunk: {e}"

@app.get("/summarize_chunks/{clerk_id}/{project_id}")
def summarize_chat_in_chunks(clerk_id: str, project_id: str):
    try:
        # Fetch full message list
        url = f"http://192.168.1.64:5000/api/v1/chats/{clerk_id}/{project_id}/executive_summary"
        response = requests.get(url)

        if response.status_code != 200:
            return {"error": f"Failed to fetch data. Status: {response.status_code}"}

        data = response.json()
        messages = data.get("message_Data", {}).get("messages", [])

        if not messages:
            return {"summary_chunks": [], "message": "No messages found."}

        # Break into chunks of 2
        summary_chunks = []
        for i in range(0, len(messages), 2):
            chunk = messages[i:i+2]
            if chunk:
                summary = summarize_chunk(chunk)
                summary_chunks.append(summary)

        return {
            "project_id": project_id,
            "clerk_id": clerk_id,
            "summary_chunks": summary_chunks,
            "total_chunks": len(summary_chunks)
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("s:app", host="0.0.0.0", port=8000, reload=True)
