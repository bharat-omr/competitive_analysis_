from openai import OpenAI
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initial setup
messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful and friendly AI business planning assistant. "
            "Ask the user one question at a time to gather their business idea. "
            "Collect details like: business type, target audience, goal, key features, and budget. "
            "After you've collected enough info (at least 4 items), STOP and say: "
            "'Thanks! Generating your business plan query now...'\n"
            "Then generate a short, single-line query like:\n"
            "'SaaS for HR teams to automate onboarding, offers payroll and document upload, budget ₹3L–₹5L.'"
        )
    },
    {
        "role": "assistant",
        "content": (
            "👋 Hello! I’m your AI business planning assistant. I’ll help you develop your business idea.\n"
            "To start, what type of business are you planning to launch?"
        )
    }
]

# Show initial assistant message
print("AI:", messages[-1]["content"])

user_inputs = []
query_generated = False

while not query_generated:
    user_input = input("You: ")
    user_inputs.append(user_input)
    messages.append({"role": "user", "content": user_input})

    # GPT response (using correct modern method)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.7
    )

    reply = response.choices[0].message.content
    print("\nAI:", reply, "\n")
    messages.append({"role": "assistant", "content": reply})

    # Stop if assistant signals query generation
    if "generating your business plan query" in reply.lower():
        query_generated = True

# Build final prompt for business query
final_prompt = (
    "From this conversation, generate a one-line structured query summarizing the business idea. "
    "Format: 'Business type for audience to achieve goal, includes features, budget ...'\n\n"
    f"User inputs: {user_inputs}"
)

# Generate final query using the assistant
final_response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": final_prompt}
    ],
    temperature=0.3
)

final_query = final_response.choices[0].message.content.strip()

# Save the query to a file
with open("business_query.txt", "w", encoding="utf-8") as f:
    f.write(final_query)

print("\n✅ Query saved to 'business_query.txt' for downstream use:\n")
print(final_query)
