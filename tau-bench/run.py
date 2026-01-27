# Copyright Sierra

import os
import argparse
import logging
import litellm
from dotenv import load_dotenv
from tau_bench.types import RunConfig
from tau_bench.run import run
from litellm import provider_list
from tau_bench.envs.user import UserStrategy

# Suppress litellm verbose logging
litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

# Load environment variables from .env file
load_dotenv()

# Drop unsupported params (e.g., temperature for GPT-5 models)
litellm.drop_params = True

# Map alternate Azure env var names to what litellm expects
if os.getenv("AZURE_SUBSCRIPTION_KEY") and not os.getenv("AZURE_API_KEY"):
    os.environ["AZURE_API_KEY"] = os.getenv("AZURE_SUBSCRIPTION_KEY")
if os.getenv("AZURE_ENDPOINT") and not os.getenv("AZURE_API_BASE"):
    os.environ["AZURE_API_BASE"] = os.getenv("AZURE_ENDPOINT")

# Map LEPTON_API_TOKEN to OPENAI_API_KEY if using a custom OpenAI-compatible endpoint
# Check if env var exists (even if empty) using "in" instead of getenv()
if "LEPTON_API_TOKEN" in os.environ and "OPENAI_API_KEY" not in os.environ:
    lepton_token = os.environ.get("LEPTON_API_TOKEN", "")
    # Use "EMPTY" placeholder if token is empty (for endpoints that don't need auth)
    os.environ["OPENAI_API_KEY"] = lepton_token if lepton_token else "EMPTY"


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument(
        "--env", type=str, choices=["retail", "airline"], default="retail"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="The model to use for the agent",
    )
    parser.add_argument(
        "--model-provider",
        type=str,
        choices=provider_list,
        help="The model provider for the agent",
    )
    parser.add_argument(
        "--user-model",
        type=str,
        default="gpt-4o",
        help="The model to use for the user simulator",
    )
    parser.add_argument(
        "--user-model-provider",
        type=str,
        choices=provider_list,
        help="The model provider for the user simulator",
    )
    parser.add_argument(
        "--agent-strategy",
        type=str,
        default="tool-calling",
        choices=["tool-calling", "act", "react", "few-shot"],
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The sampling temperature for the action model",
    )
    parser.add_argument(
        "--task-split",
        type=str,
        default="test",
        choices=["train", "test", "dev"],
        help="The split of tasks to run (only applies to the retail domain for now",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--end-index", type=int, default=-1, help="Run all tasks if -1"
    )
    parser.add_argument(
        "--task-ids",
        type=int,
        nargs="+",
        help="(Optional) run only the tasks with the given IDs",
    )
    parser.add_argument("--log-dir", type=str, default="results")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Number of tasks to run in parallel",
    )
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--shuffle", type=int, default=0)
    parser.add_argument(
        "--user-strategy",
        type=str,
        default="llm",
        choices=[item.value for item in UserStrategy],
    )
    parser.add_argument(
        "--few-shot-displays-path",
        type=str,
        help="Path to a jsonlines file containing few shot displays",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="Base URL for the agent model API (for OpenAI-compatible endpoints)",
    )
    parser.add_argument(
        "--user-api-base",
        type=str,
        default=None,
        help="Base URL for the user model API (for OpenAI-compatible endpoints)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the agent model (sets OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--user-api-key",
        type=str,
        default=None,
        help="API key for the user model",
    )
    args = parser.parse_args()

    # Set API keys from command line args if provided
    # Use "EMPTY" as placeholder if api_key is empty string (for endpoints that don't need auth)
    if args.api_key is not None:
        os.environ["OPENAI_API_KEY"] = (
            args.api_key if args.api_key else "EMPTY"
        )
    if args.user_api_key:
        # For user model, set appropriate env var based on provider
        if args.user_model_provider == "azure":
            os.environ["AZURE_API_KEY"] = args.user_api_key
        else:
            os.environ["OPENAI_API_KEY"] = args.user_api_key

    # For Azure provider, default to AZURE_API_BASE env var if api_base not provided
    api_base = args.api_base
    user_api_base = args.user_api_base
    azure_api_base = os.getenv("AZURE_API_BASE")

    if args.model_provider == "azure" and not api_base and azure_api_base:
        api_base = azure_api_base
    if (
        args.user_model_provider == "azure"
        and not user_api_base
        and azure_api_base
    ):
        user_api_base = azure_api_base

    print(args)
    return RunConfig(
        model_provider=args.model_provider,
        user_model_provider=args.user_model_provider,
        model=args.model,
        user_model=args.user_model,
        api_base=api_base,
        user_api_base=user_api_base,
        num_trials=args.num_trials,
        env=args.env,
        agent_strategy=args.agent_strategy,
        temperature=args.temperature,
        task_split=args.task_split,
        start_index=args.start_index,
        end_index=args.end_index,
        task_ids=args.task_ids,
        log_dir=args.log_dir,
        max_concurrency=args.max_concurrency,
        seed=args.seed,
        shuffle=args.shuffle,
        user_strategy=args.user_strategy,
        few_shot_displays_path=args.few_shot_displays_path,
    )


def main():
    config = parse_args()
    run(config)


if __name__ == "__main__":
    main()
