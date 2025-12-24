import copy
import time

import httpx
from anthropic import AsyncAnthropicVertex
from google import genai
from google.genai import types
from google.genai.types import Content, GenerateContentConfig, Part
from groq import AsyncGroq
from openai import AsyncAzureOpenAI, AsyncOpenAI


async def call_sarvam_model(messages, model, model_url, llm_config, api_key=None):
    headers = {"Content-Type": "application/json"}

    # Add Authorization header if API key is provided
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = copy.deepcopy(llm_config)
    payload["stream"] = False
    payload["messages"] = messages
    payload["model"] = model
    # payload["chat_template_kwargs"] = {"enable_thinking": False}
    start_time = time.time()
    async with httpx.AsyncClient() as client:
        response = await client.post(
            url=model_url,
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
    try:
        model_output = response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error calling Sarvam model: {str(e)}")
        model_output = None
    end_time = time.time()
    return model_output, end_time - start_time


async def call_azure_model(messages, deployment, endpoint, api_key, llm_config):
    start_time = time.time()
    try:
        client = AsyncAzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version="2024-12-01-preview",
        )

        completion = await client.chat.completions.create(
            model=deployment,
            messages=messages,
            # max_tokens=llm_config.get("max_tokens", 512),
            # temperature=llm_config.get("temperature", 0.1),
            stream=False,
        )
        end_time = time.time()
        response_content = completion.choices[0].message.content
        return response_content, end_time - start_time
    except Exception as e:
        end_time = time.time()
        print(f"Error calling Azure model: {str(e)}")
        return None, end_time - start_time


async def call_openai_model(messages, model, api_key, llm_config):
    start_time = time.time()
    try:
        kwargs = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if model == "gpt-4o" or model == "gpt-4o-mini":
            kwargs["max_tokens"] = llm_config.get("max_tokens", 512)
            kwargs["temperature"] = llm_config.get("temperature", 0.1)
        elif model == "o4-mini" or model == "o3" or model == "gpt-5":
            kwargs["reasoning_effort"] = llm_config.get("reasoning_effort", "medium")

        client = AsyncOpenAI(api_key=api_key)
        completion = await client.chat.completions.create(**kwargs)
        response_content = completion.choices[0].message.content
        end_time = time.time()
        return response_content, end_time - start_time

    except Exception as e:
        end_time = time.time()
        print(f"Error calling OpenAI model: {str(e)}")
        return None, end_time - start_time


async def call_anthropic_vertex_model(messages, model, project_id, region, llm_config):
    start_time = time.time()
    try:
        # Local import to avoid hard dependency during tests when unused

        # Extract system prompt if present. Do not mutate roles here; that is handled earlier
        system_prompt = None
        transformed_messages = []
        if messages and messages[0].get("role") == "system":
            system_prompt = messages[0].get("content", None)
            remaining = messages[1:]
        else:
            remaining = messages

        for m in remaining:
            transformed_messages.append(
                {"role": m.get("role"), "content": m.get("content", "")}
            )

        client = AsyncAnthropicVertex(project_id=project_id, region=region)

        completion = await client.messages.create(
            model=model,
            system=system_prompt,
            messages=transformed_messages,
            max_tokens=llm_config.get("max_tokens", 512),
            temperature=llm_config.get("temperature", 0.1),
        )
        end_time = time.time()

        # Concatenate any text content blocks
        text = "".join(
            [getattr(block, "text", "") for block in getattr(completion, "content", [])]
        )
        return text, end_time - start_time
    except Exception as e:
        end_time = time.time()
        print(f"Error calling Anthropic Vertex model: {str(e)}")
        return None, end_time - start_time


async def call_gemini_model(messages, model, api_key, llm_config):
    start_time = time.time()
    try:

        client = genai.Client(api_key=api_key)

        def map_role_for_gemini(role: str) -> str:
            if role in ["user", "system", "tool"]:
                return "user"
            elif role in ["assistant", "model"]:
                return "model"
            else:
                return "user"

        contents = [
            Content(
                role=map_role_for_gemini(m.get("role", "user")),
                parts=[Part(text=m.get("content", ""))],
            )
            for m in messages
        ]

        config = GenerateContentConfig(
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(include_thoughts=False),
            # temperature=llm_config.get("temperature", 0.1),
            # max_output_tokens=llm_config.get("max_tokens", 512),
        )

        response = await client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        end_time = time.time()

        text = None
        try:
            if response and getattr(response, "candidates", None):
                cand0 = response.candidates[0]
                if cand0 and getattr(cand0, "content", None):
                    parts = cand0.content.parts
                    if (
                        parts
                        and len(parts) > 0
                        and getattr(parts[0], "text", None) is not None
                    ):
                        text = parts[0].text
            if not text:
                text = getattr(response, "text", None)
        except Exception:
            text = getattr(response, "text", None)

        return text, end_time - start_time
    except Exception as e:
        end_time = time.time()
        print(f"Error calling Gemini model: {str(e)}")
        return None, end_time - start_time


async def call_groq_model(messages, model, api_key, llm_config):
    start_time = time.time()
    try:
        client = AsyncGroq(api_key=api_key)

        completion = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=llm_config.get("max_tokens", 512),
            temperature=llm_config.get("temperature", 0.1),
            reasoning_effort="medium",
            stream=False,
        )
        end_time = time.time()
        content = completion.choices[0].message.content
        return content, end_time - start_time
    except Exception as e:
        end_time = time.time()
        print(f"Error calling Groq model: {str(e)}")
        return None, end_time - start_time
