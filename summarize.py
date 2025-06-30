from fastapi import FastAPI
import requests
import google.generativeai as genai
import os
from dotenv import load_dotenv
import uvicorn

# Load environment variables
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
You are a helpful AI. Summarize the following short chat exchange in 1-2 sentences:

{chat_text}
"""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Error summarizing chunk: {e}"


@app.put("/summarize_and_save/{clerk_id}/{project_id}")
def summarize_and_save(clerk_id: str, project_id: str):
    try:
        # Step 1: Fetch chat messages
        fetch_url = f"http://192.168.1.64:5000/api/v1/chats/{clerk_id}/{project_id}/executive_summary"
        response = requests.get(fetch_url)

        if response.status_code != 200:
            return {"error": f"Failed to fetch data. Status: {response.status_code}"}

        data = response.json()
        messages = data.get("message_Data", {}).get("messages", [])
        if not messages:
            return {"summary_chunks": [], "message": "No messages found."}

        # Step 2: Summarize in chunks of 2
        summary_chunks = []
        for i in range(0, len(messages), 2):
            chunk = messages[i:i+2]
            if chunk:
                summary = summarize_chunk(chunk)
                summary_chunks.append(summary)

        # Step 3: Prepare and send payload to save API via PUT
        save_url = f"http://192.168.1.64:5000/api/v1/chats/save-type-summary/{clerk_id}/{project_id}/executive_summary"
        save_payload = {
            "content": " ".join(summary_chunks)
        }

        save_response = requests.put(save_url, json=save_payload)

        if save_response.status_code != 200:
            return {
                "error": f"Failed to save summaries. Status: {save_response.status_code}",
                "response_text": save_response.text,
                "request_payload": save_payload,
                "url": save_url
            }

        return {
            "project_id": project_id,
            "clerk_id": clerk_id,
            "summary_chunks": summary_chunks,
            "status": "✅ Summaries saved successfully."
        }

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run("summarize:app", host="0.0.0.0", port=9000, reload=True)
