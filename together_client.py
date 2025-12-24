from together import Together

TOGETHER_API_KEY="6e98df5e19cc521af9d51f24c7c2a15c2b452137ffa8dfd6d17ad5cb87a9c699" #noqa: E501

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
