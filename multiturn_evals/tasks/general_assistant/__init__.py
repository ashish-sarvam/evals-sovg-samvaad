"""
General Assistant Task

Tests: Does the assistant correctly help users with everyday queries
in their language?
"""

from tasks import BaseTask, TaskConfig
from core.languages import Language, SUPPORTED_LANGUAGES

from tasks.general_assistant.users import (
    get_user_prompts,
    list_users,
    list_agents,
)
from tasks.general_assistant.verifier import (
    VERIFIER_SYSTEM_PROMPT,
    VERIFIER_USER_TEMPLATE,
)


class GeneralAssistantTask(BaseTask):
    """Tests agent's ability to help users with general queries."""

    def __init__(self, agent: str, user: str):
        self.config = TaskConfig(
            name="general_assistant",
            description="Test assistant helpfulness in multiple languages",
            languages=SUPPORTED_LANGUAGES.copy(),
            agent_name=agent,
        )
        self.agent = agent
        self.user = user

        # Validate agent and user
        user_prompts = get_user_prompts(agent)
        if user not in user_prompts:
            available = ", ".join(user_prompts.keys())
            raise ValueError(f"Unknown user: {user}. Available: {available}")

        self._user_prompts = user_prompts

    def get_user_prompt(self, language: Language) -> str:
        prompt_lang = language.prompt_language
        return self._user_prompts[self.user].format(LANGUAGE=prompt_lang)

    def get_verifier_system_prompt(self) -> str:
        return VERIFIER_SYSTEM_PROMPT

    def get_verifier_user_prompt(
        self, conversation: list[dict], language: Language
    ) -> str:
        conv_text = "\n".join(
            f"[{m['role'].upper()}]: {m['content']}" for m in conversation
        )
        return VERIFIER_USER_TEMPLATE.format(
            language=language.name, conversation=conv_text
        )

    @staticmethod
    def list_users(agent: str) -> list[str]:
        return list_users(agent)

    @staticmethod
    def list_agents() -> list[str]:
        return list_agents()
