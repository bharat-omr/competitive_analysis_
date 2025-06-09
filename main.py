import os
from dotenv import load_dotenv
from langchain_community.utilities import SerpAPIWrapper
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()
os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize web search and generative model
search = SerpAPIWrapper()
gemini_model = genai.GenerativeModel("gemini-2.0-flash")  # or use "gemini-pro" if needed

# Take user input
user_input = input("Enter your business question for competitive analysis: ")

# Run real-time web search
serp_result = search.run(user_input)

# Rewritten prompt tailored for competitive analysis
web_prompt = f"""
You are an expert AI market analyst. Using the real-time web search result below and your own business knowledge, perform a detailed competitive analysis.

Task:
- Analyze the competitive landscape relevant to the query.
- Identify key competitors, their strengths, weaknesses, and strategies.
- Summarize any current trends, opportunities, and risks.
- Provide actionable insights for a new or existing business.

User Query: {user_input}
Web Search Result: {serp_result}

Answer:
"""

# Generate response from Gemini
gemini_response = gemini_model.generate_content(web_prompt)

# Display the response
answer = f"📡 *Competitive Analysis with Web Support*\n\n{gemini_response.text.strip()}"
print(answer)
