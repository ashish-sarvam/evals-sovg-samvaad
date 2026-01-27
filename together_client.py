import os
from together import Together
from dotenv import load_dotenv

load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")

client = Together(api_key=TOGETHER_API_KEY)
stream = client.chat.completions.create(
    model="zai-org/GLM-4.5-Air-FP8",
    messages=[
        {
            "role": "user",
            "content": "What are the top 3 things to do in New York?",
        }
    ],
    stream=True,
)

for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
