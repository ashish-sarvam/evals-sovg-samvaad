"""
Conversation runner infrastructure.

Handles the mechanics of running conversations between agent and user models.
This is task-agnostic - tasks define WHAT to test, runner handles HOW.
"""

import re
from typing import TYPE_CHECKING

from core.languages import Language
from core.config import EVAL_DEFAULTS


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from text.

    Also handles unclosed <think> tags by removing from <think> to end.
    """
    # First, remove complete <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Then, remove unclosed <think> tags (from <think> to end of string)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return text.strip()


if TYPE_CHECKING:
    from tasks import BaseTask
    from core.models import LLMModel


def build_agent_messages(agent_module, language: Language, task=None) -> list[dict]:
    """Build initial messages for the agent with language injected."""
    system_prompt = agent_module.SYSTEM_PROMPT.replace(
        "{LANGUAGE}", language.prompt_language
    )

    # Inject tone prompt if task has one (e.g., colloquial task)
    if task and hasattr(task, "get_tone_prompt"):
        tone_prompt = task.get_tone_prompt(language)
        if tone_prompt:
            # Insert tone prompt after the first line (agent identity)
            lines = system_prompt.split("\n", 1)
            if len(lines) > 1:
                system_prompt = lines[0] + "\n" + tone_prompt + "\n" + lines[1]
            else:
                system_prompt = system_prompt + "\n" + tone_prompt

    # Inject personalization prompt if task has one (e.g., memory task)
    if task and hasattr(task, "get_personalization_prompt"):
        personalization_prompt = task.get_personalization_prompt(language)
        if personalization_prompt:
            # Insert personalization prompt before GUIDELINES section
            if "## GUIDELINES" in system_prompt:
                system_prompt = system_prompt.replace(
                    "## GUIDELINES",
                    personalization_prompt + "\n\n---\n\n## GUIDELINES",
                )
            else:
                # If no GUIDELINES section, append at the end
                system_prompt = system_prompt + "\n\n---\n\n" + personalization_prompt

    first_user_msg = getattr(
        agent_module, "FIRST_USER_MESSAGE", "Start the conversation."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": first_user_msg},
    ]


class ConversationRunner:
    """
    Runs conversations between an agent and user proxy.

    The runner is task-agnostic - it receives task-specific behavior
    through the task object (user prompts, stop conditions, etc.)
    """

    def __init__(
        self,
        agent_model: "LLMModel",
        user_model: "LLMModel",
        task: "BaseTask",
        verbose: bool = False,
    ):
        self.agent_model = agent_model
        self.user_model = user_model
        self.task = task
        self.verbose = verbose

    def run(self, agent_module, language: Language) -> list[dict]:
        """
        Run a complete conversation.

        Args:
            agent_module: The agent configuration module
            language: Target language for the conversation

        Returns:
            List of conversation messages (role: agent/user, content: str)
        """
        agent_thread = build_agent_messages(agent_module, language, self.task)
        user_thread = [
            {"role": "system", "content": self.task.get_user_prompt(language)}
        ]

        conversation: list[dict] = []
        max_turns = self.task.config.get("max_turns", EVAL_DEFAULTS["max_turns"])

        if self.verbose:
            self._print(f"\n{'─'*60}")
            self._print(f"Starting: {language.name}")
            self._print(f"{'─'*60}")

        # Initial agent response
        if self.verbose:
            self._print(f"[Turn 0] Agent responding...")
        agent_response = self.agent_model.get_response(agent_thread)
        agent_response = strip_think_tags(agent_response)  # Remove <think> tags
        agent_thread.append({"role": "assistant", "content": agent_response})
        conversation.append({"role": "agent", "content": agent_response})

        if self.verbose:
            self._print(f"[AGENT]: {agent_response}")

        # Conversation loop
        for turn in range(max_turns):
            # Show progress (even without verbose)
            print(f"T{turn + 1}", end=".", flush=True)

            # Build user context
            user_context = self._build_user_context(user_thread, conversation)

            # Get user response
            if self.verbose:
                self._print(f"\n[Turn {turn + 1}] User responding...")
            user_response = self.user_model.get_response(user_context)

            if self.verbose:
                self._print(f"[USER]: {user_response}")

            # Check for stop signal - flexible detection
            if self._has_stop_signal(user_response):
                clean_response = self._remove_stop_signal(user_response)
                if clean_response:
                    conversation.append({"role": "user", "content": clean_response})
                if self.verbose:
                    self._print("[END - stop signal]")
                print(" STOP", end=" ", flush=True)
                break

            # Check for natural end
            if self._is_conversation_ending(agent_response):
                conversation.append({"role": "user", "content": user_response})
                if self.verbose:
                    self._print("[END - natural conclusion]")
                print(" END", end=" ", flush=True)
                break

            conversation.append({"role": "user", "content": user_response})
            agent_thread.append({"role": "user", "content": user_response})

            # Get agent response
            if self.verbose:
                self._print(f"\n[Turn {turn + 1}] Agent responding...")
            agent_response = self.agent_model.get_response(agent_thread)
            agent_response = strip_think_tags(agent_response)  # Remove <think> tags
            agent_thread.append({"role": "assistant", "content": agent_response})
            conversation.append({"role": "agent", "content": agent_response})

            if self.verbose:
                self._print(f"[AGENT]: {agent_response}")

        if self.verbose:
            self._print(f"{'─'*60}")
            self._print(f"Completed: {len(conversation)} messages")

        return conversation

    def _build_user_context(
        self, user_thread: list[dict], conversation: list[dict]
    ) -> list[dict]:
        """Build the context for user model to generate response."""
        context = user_thread.copy()

        # Add conversation from user's perspective (agent messages are "user" to the user model)
        for msg in conversation:
            if msg["role"] == "agent":
                context.append({"role": "user", "content": msg["content"]})
            else:
                context.append({"role": "assistant", "content": msg["content"]})

        context.append(
            {
                "role": "user",
                "content": "Generate your next response. Keep it natural and brief.",
            }
        )
        return context

    def _has_stop_signal(self, response: str) -> bool:
        """Check if response contains stop signal (flexible detection)."""
        # Check various formats of STOP
        stop_patterns = [
            "**STOP**",
            "**stop**",
            "*STOP*",
            "STOP",
            "[STOP]",
            "(STOP)",
        ]
        response_upper = response.upper()
        for pattern in stop_patterns:
            if pattern.upper() in response_upper:
                return True
        return False

    def _remove_stop_signal(self, response: str) -> str:
        """Remove stop signal from response."""
        import re

        # Remove various STOP formats
        cleaned = response
        patterns = [
            r"\*\*STOP\*\*",
            r"\*STOP\*",
            r"\[STOP\]",
            r"\(STOP\)",
            r"\bSTOP\b",
        ]
        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def _is_conversation_ending(self, agent_response: str) -> bool:
        """Check if agent response indicates conversation is ending."""
        end_indicators = ["धन्यवाद", "शुभ दिन", "nice day", "goodbye", "have a good"]
        response_lower = agent_response.lower()
        return any(indicator in response_lower for indicator in end_indicators)

    def _print(self, msg: str):
        """Print with consistent formatting."""
        print(msg)
