"""
Generate Transcripts - Interactive chat with model(s).

Usage:
    poetry run python generate_transcript.py [agent] [options]

Examples:
    poetry run python generate_transcript.py idfc -l Hindi
    poetry run python generate_transcript.py idfc -m1 tinker
    poetry run python generate_transcript.py idfc -m1 tinker -m2 azure2

Arguments:
    agent               Optional. Agent name (e.g., idfc). Prompts if not given.

Options:
    -l, --language      Optional. Output language. Default: Hindi
    -m1, --model1       Optional. Model 1 backend. Default: azure1
    -m2, --model2       Optional. Model 2 backend. If not provided, single model mode.
    -t, --translate     Optional. Show English translations.

Models: azure1, azure2, tinker
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from openai import AzureOpenAI
from agents import get_agent, list_agents
from typing import Optional
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv(Path(__file__).parent.parent / ".env")

# Directory for saving transcripts
TRANSCRIPTS_DIR = Path(__file__).parent / "artifacts" / "transcripts"

endpoint = os.getenv("AZURE_ENDPOINT", "")
api_version = "2024-12-01-preview"
subscription_key = os.getenv("AZURE_API_KEY", "")

# Azure Model configurations
MODEL_1_DEPLOYMENT = "1-mini-2025-04-14-3la"  # Finetuned model
MODEL_2_DEPLOYMENT = "gpt-4.1-mini"  # Non-finetuned model
TEMPERATURE = 0.1

azure_client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

# Tinker model (lazy loaded)
_tinker_model = None


def get_tinker_model():
    """Lazy load Tinker model to avoid import overhead if not used."""
    global _tinker_model
    if _tinker_model is None:
        from tinker_helper import TinkerModel
        _tinker_model = TinkerModel(temperature=TEMPERATURE)
    return _tinker_model


# Model name mapping
MODEL_NAMES = {
    "azure1": MODEL_1_DEPLOYMENT,
    "azure2": MODEL_2_DEPLOYMENT,
    "tinker": "tinker-finetuned",
}


def get_response(messages: list, model_type: str) -> str:
    """
    Get response from the specified model.

    Args:
        messages: List of message dicts
        model_type: One of "azure1", "azure2", or "tinker"
    """
    if model_type == "tinker":
        tinker_model = get_tinker_model()
        # print("messages", messages)
        return tinker_model.get_response(messages)
    else:
        # Azure models
        deployment = MODEL_1_DEPLOYMENT if model_type == "azure1" else MODEL_2_DEPLOYMENT
        response = azure_client.chat.completions.create(
            messages=messages,
            max_completion_tokens=13107,
            temperature=1.0,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            model=deployment,
        )
        return response.choices[0].message.content or ""


def print_separator(char: str = "=", length: int = 60):
    print(char * length)


def translate_to_english(text: str) -> str:
    """Translate text to English using Azure model 2."""
    response = azure_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a translator. Translate the following text to "
                    "English. Preserve the meaning and tone. Output only the "
                    "translation, nothing else."
                ),
            },
            {"role": "user", "content": text},
        ],
        max_completion_tokens=2000,
        temperature=0.3,
        model=MODEL_2_DEPLOYMENT,
    )
    return response.choices[0].message.content or ""


def select_agent(agent_arg: str | None = None):
    """Let user select an agent from available options."""
    available = list_agents()

    if agent_arg:
        # Agent specified via command line
        if agent_arg in available:
            return get_agent(agent_arg)
        print(f"Unknown agent: {agent_arg}")

    print_separator()
    print("Available Agents:")
    for i, name in enumerate(available, 1):
        agent = get_agent(name)
        print(f"  {i}. {name} - {agent.AGENT_NAME}")
    print_separator()

    while True:
        try:
            choice = input("Select agent (number or name): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            sys.exit(0)

        # Try as number
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(available):
                return get_agent(available[idx])

        # Try as name
        if choice in available:
            return get_agent(choice)

        print(f"Invalid choice. Enter 1-{len(available)} or agent name.")


def build_initial_messages(agent, language: str) -> list:
    """Build initial message thread from agent config with language injected."""
    # Use replace instead of format to avoid issues with JSON braces in prompt
    system_prompt = agent.SYSTEM_PROMPT.replace("{LANGUAGE}", language)
    
    # Get custom first user message if defined, otherwise use default
    first_user_msg = getattr(agent, "FIRST_USER_MESSAGE", "Start the conversation.")
    
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": first_user_msg,
        },
    ]


def save_transcripts(
    agent_name: str,
    model_1_thread: list,
    model_2_thread: Optional[list],
    language: str,
    model_1_type: str,
    model_2_type: Optional[str],
):
    """Save conversation transcripts to JSON files."""
    TRANSCRIPTS_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{agent_name}_{language}_{timestamp}"

    # Filter out system messages for cleaner transcripts
    def clean_thread(thread):
        return [msg for msg in thread if msg["role"] != "system"]

    # Save model 1 transcript
    model_1_file = TRANSCRIPTS_DIR / f"{base_name}_{model_1_type}.json"
    with open(model_1_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "agent": agent_name,
                "model": model_1_type,
                "deployment": MODEL_NAMES.get(model_1_type, model_1_type),
                "language": language,
                "timestamp": timestamp,
                "messages": clean_thread(model_1_thread),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved model 1 transcript: {model_1_file}")

    # Save model 2 transcript only if model 2 was used
    if model_2_type and model_2_thread is not None:
        model_2_file = TRANSCRIPTS_DIR / f"{base_name}_{model_2_type}.json"
        with open(model_2_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "agent": agent_name,
                    "model": model_2_type,
                    "deployment": MODEL_NAMES.get(model_2_type, model_2_type),
                    "language": language,
                    "timestamp": timestamp,
                    "messages": clean_thread(model_2_thread),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Saved model 2 transcript: {model_2_file}")


def generate_and_display_responses(
    model_1_thread: list,
    model_2_thread: Optional[list],
    model_1_type: str,
    model_2_type: Optional[str],
    include_translation: bool = False,
):
    """Generate responses from model(s) in parallel and optionally display translations."""
    model_1_response = None
    model_2_response = None
    model_1_error = None
    model_2_error = None

    # Run model calls in parallel if both models are specified
    if model_2_type and model_2_thread is not None:
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_1 = executor.submit(get_response, model_1_thread, model_1_type)
            future_2 = executor.submit(get_response, model_2_thread, model_2_type)

            try:
                model_1_response = future_1.result()
            except Exception as e:
                model_1_error = e

            try:
                model_2_response = future_2.result()
            except Exception as e:
                model_2_error = e
    else:
        # Single model mode
        try:
            model_1_response = get_response(model_1_thread, model_1_type)
        except Exception as e:
            model_1_error = e

    # Display model 1 results
    print_separator("-")
    print(f"[MODEL 1 ({model_1_type})]")
    if model_1_error:
        print(f"Error: {model_1_error}")
    elif model_1_response:
        print(model_1_response)
        model_1_thread.append({"role": "assistant", "content": model_1_response})

    if include_translation and model_1_response:
        print_separator("-")
        print("[MODEL 1 - ENGLISH TRANSLATION]")
        try:
            model_1_english = translate_to_english(model_1_response)
            print(model_1_english)
        except Exception as e:
            print(f"Error translating: {e}")

    # Display model 2 results if model 2 was used
    if model_2_type and model_2_thread is not None:
        print_separator("-")
        print(f"[MODEL 2 ({model_2_type})]")
        if model_2_error:
            print(f"Error: {model_2_error}")
        elif model_2_response:
            print(model_2_response)
            model_2_thread.append({"role": "assistant", "content": model_2_response})

        if include_translation and model_2_response:
            print_separator("-")
            print("[MODEL 2 - ENGLISH TRANSLATION]")
            try:
                model_2_english = translate_to_english(model_2_response)
                print(model_2_english)
            except Exception as e:
                print(f"Error translating: {e}")

    print_separator("-")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate transcripts comparing two models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_transcript.py idfc --language Hindi --translate
  python generate_transcript.py idfc -l Tamil -t
  python generate_transcript.py idfc --model1 tinker --model2 azure2
  python generate_transcript.py idfc -m1 azure1 -m2 tinker

Model options: azure1 (finetuned), azure2 (gpt-4.1-mini), tinker (tinker finetuned)
        """,
    )
    parser.add_argument(
        "agent",
        nargs="?",
        default=None,
        help="Agent name (e.g., idfc). If not provided, will prompt for selection.",
    )
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        default=None,
        help="Output language for the agent (e.g., Tamil, Hindi, Telugu). "
        "Defaults to Hindi.",
    )
    parser.add_argument(
        "-t",
        "--translate",
        action="store_true",
        help="Include English translations of responses",
    )
    parser.add_argument(
        "-m1",
        "--model1",
        type=str,
        choices=["azure1", "azure2", "tinker"],
        default="azure1",
        help="Model 1 backend: azure1, azure2, or tinker (default: azure1)",
    )
    parser.add_argument(
        "-m2",
        "--model2",
        type=str,
        choices=["azure1", "azure2", "tinker"],
        default=None,
        help="Model 2 backend: azure1, azure2, or tinker (optional)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Select agent
    agent = select_agent(args.agent)
    # Get module name (e.g., "idfc")
    agent_name = agent.__name__.split(".")[-1]
    print(f"\nUsing agent: {agent.AGENT_NAME}")

    # Determine language: CLI arg > fallback to Hindi
    language = args.language if args.language else "Hindi"
    model_1_type = args.model1
    model_2_type = args.model2  # Can be None

    print(f"Language: {language}")
    print(f"Translation: {'Enabled' if args.translate else 'Disabled'}")

    # Initialize threads with language injected into system prompt
    initial_messages = build_initial_messages(agent, language)
    model_1_thread = [msg.copy() for msg in initial_messages]
    model_2_thread = [msg.copy() for msg in initial_messages] if model_2_type else None

    print_separator()
    if model_2_type:
        print("Interactive Chat - Comparing Model 1 vs Model 2")
        m1_name = MODEL_NAMES.get(model_1_type, model_1_type)
        m2_name = MODEL_NAMES.get(model_2_type, model_2_type)
        print(f"  Model 1: {model_1_type} ({m1_name})")
        print(f"  Model 2: {model_2_type} ({m2_name})")
    else:
        print("Interactive Chat - Single Model")
        m1_name = MODEL_NAMES.get(model_1_type, model_1_type)
        print(f"  Model: {model_1_type} ({m1_name})")
    print_separator()
    print("\nType 'quit' or 'exit' to end the conversation.")
    print("Type 'history' to see the conversation thread.")
    print_separator()

    # Generate initial assistant responses
    print("\nGenerating initial responses...")
    generate_and_display_responses(
        model_1_thread, model_2_thread,
        model_1_type, model_2_type,
        include_translation=args.translate
    )

    while True:
        try:
            user_input = input("\n[You]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nExiting chat...")
            break

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit"]:
            print("\nExiting chat...")
            break

        if user_input.lower() == "history":
            print_separator("-")
            print(f"\n[MODEL 1 THREAD ({model_1_type})]")
            for msg in model_1_thread:
                if msg["role"] != "system":
                    print(f"  {msg['role'].upper()}: {msg['content'][:100]}...")
            if model_2_type and model_2_thread:
                print(f"\n[MODEL 2 THREAD ({model_2_type})]")
                for msg in model_2_thread:
                    if msg["role"] != "system":
                        print(f"  {msg['role'].upper()}: {msg['content'][:100]}...")
            print_separator("-")
            continue

        # Add user message to thread(s)
        user_message = {"role": "user", "content": user_input}
        model_1_thread.append(user_message.copy())
        if model_2_thread is not None:
            model_2_thread.append(user_message.copy())

        # Get responses from model(s)
        print("\nGenerating responses...")
        generate_and_display_responses(
            model_1_thread, model_2_thread,
            model_1_type, model_2_type,
            include_translation=args.translate
        )

    # Save transcripts on exit
    print("\nSaving transcripts...")
    save_transcripts(
        agent_name, model_1_thread, model_2_thread,
        language, model_1_type, model_2_type
    )


if __name__ == "__main__":
    main()
