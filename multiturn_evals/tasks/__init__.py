"""
Evaluation tasks.

Each task is a folder containing:
- __init__.py  - Task class
- user_prompt.py - User behavior prompt
- verifier.py - Verification prompts and criteria
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json

from core.languages import Language, SUPPORTED_LANGUAGES
from core.config import EVAL_DEFAULTS


@dataclass
class TaskConfig:
    """Configuration for an evaluation task."""

    name: str
    description: str
    languages: list[Language] = field(
        default_factory=lambda: SUPPORTED_LANGUAGES.copy()
    )
    max_turns: int = EVAL_DEFAULTS["max_turns"]
    agent_name: str = EVAL_DEFAULTS["agent_name"]
    verifier_provider: str = EVAL_DEFAULTS["verifier_provider"]

    def get(self, key: str, default=None):
        return getattr(self, key, default)


class BaseTask(ABC):
    """
    Base class for evaluation tasks.

    Each task defines:
    - User behavior via prompts
    - Verification criteria and prompts
    """

    config: TaskConfig

    @abstractmethod
    def get_user_prompt(self, language: Language) -> str:
        """Get user proxy system prompt for given language."""
        pass

    @abstractmethod
    def get_verifier_system_prompt(self) -> str:
        """Get system prompt for verification LLM."""
        pass

    @abstractmethod
    def get_verifier_user_prompt(
        self, conversation: list[dict], language: Language
    ) -> str:
        """Get user prompt for verification with conversation data."""
        pass

    def verify(
        self, conversation: list[dict], language: Language, verifier_model
    ) -> dict:
        """Verify a conversation using the verifier model."""
        messages = [
            {"role": "system", "content": self.get_verifier_system_prompt()},
            {
                "role": "user",
                "content": self.get_verifier_user_prompt(conversation, language),
            },
        ]

        try:
            response = verifier_model.get_response(messages)
            return self._parse_verification_response(response)
        except Exception as e:
            return {
                "overall_status": "error",
                "overall_score": 0.0,
                "error": str(e),
            }

    def _parse_verification_response(self, response: str) -> dict:
        """Parse JSON response from verifier (boolean format + optional scores)."""
        try:
            text = response
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            data = json.loads(text.strip())

            results = []
            scores = {}  # For numeric scores like colloquial_score
            pass_count = 0
            total_count = 0

            for key, value in data.items():
                if key == "summary":
                    continue  # Handle summary separately

                if isinstance(value, dict):
                    # Boolean check (has "result" field)
                    if "result" in value:
                        is_pass = value.get("result", False)
                        total_count += 1
                        if is_pass:
                            pass_count += 1
                        result_entry = {
                            "check_name": key,
                            "passed": is_pass,
                            "reason": value.get("reason", ""),
                        }
                        # Add snippet if present and not empty
                        snippet = value.get("snippet", "")
                        if snippet:
                            result_entry["snippet"] = snippet
                        results.append(result_entry)

                    # Numeric score (has "score" field)
                    elif "score" in value:
                        score_entry = {
                            "check_name": key,
                            "score": value.get("score", 0),
                            "reason": value.get("reason", ""),
                        }
                        if "expected" in value:
                            score_entry["expected"] = value["expected"]
                        if "examples" in value:
                            score_entry["examples"] = value["examples"]
                        scores[key] = score_entry

            # Calculate overall
            if total_count == 0:
                overall_status = "error"
                pass_rate = 0.0
            elif pass_count == total_count:
                overall_status = "pass"
                pass_rate = 1.0
            elif pass_count == 0:
                overall_status = "fail"
                pass_rate = 0.0
            else:
                overall_status = "partial"
                pass_rate = pass_count / total_count

            result = {
                "overall_status": overall_status,
                "pass_rate": pass_rate,
                "passed": pass_count,
                "total": total_count,
                "results": results,
                "summary": data.get("summary", ""),
                "raw_response": response,
            }

            # Add scores if present
            if scores:
                result["scores"] = scores

            return result

        except json.JSONDecodeError as e:
            return {
                "overall_status": "error",
                "pass_rate": 0.0,
                "error": f"JSON parse error: {e}",
                "raw_response": response,
            }


# =============================================================================
# TASK REGISTRY
# =============================================================================

# Import after BaseTask is defined to avoid circular imports
from tasks.multilingual import MultilingualTask
from tasks.english_user import EnglishUserTask
from tasks.colloquial import ColloquialTask
from tasks.roman_user import RomanUserTask
from tasks.conversationality import ConversationalityTask
from tasks.robustness import RobustnessTask
from tasks.general_assistant import GeneralAssistantTask
from tasks.memory import MemoryTask

TASK_REGISTRY: dict[str, type[BaseTask]] = {
    "multilingual": MultilingualTask,
    "english_user": EnglishUserTask,
    "colloquial": ColloquialTask,
    "roman_user": RomanUserTask,
    "conversationality": ConversationalityTask,
    "robustness": RobustnessTask,
    "general_assistant": GeneralAssistantTask,
    "memory": MemoryTask,
}


def get_task(name: str, agent: str, user: str) -> BaseTask:
    """Get a task instance by name, agent, and user type."""
    if name not in TASK_REGISTRY:
        available = ", ".join(TASK_REGISTRY.keys())
        raise ValueError(f"Unknown task: {name}. Available: {available}")
    return TASK_REGISTRY[name](agent=agent, user=user)


def list_tasks() -> list[str]:
    """List available task names."""
    return list(TASK_REGISTRY.keys())


def list_agents(task_name: str) -> list[str]:
    """List available agents for a task."""
    if task_name not in TASK_REGISTRY:
        return []
    return TASK_REGISTRY[task_name].list_agents()


def list_users(task_name: str, agent: str) -> list[str]:
    """List available users for a task and agent."""
    if task_name not in TASK_REGISTRY:
        return []
    return TASK_REGISTRY[task_name].list_users(agent)
