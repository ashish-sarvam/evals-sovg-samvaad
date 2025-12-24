from openai import OpenAI

# Replace with your actual token
client = OpenAI(
    base_url="https://h1v6kgoi-qwen3-30b-a3b-tools.xenon.lepton.run/v1",
    api_key=""
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-30B-A3B-Instruct-2507",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum entanglement simply."}
    ]
)

print(response.choices[0].message.content)