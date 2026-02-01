"""
Memory/Personalization Task

Tests: Does the agent appropriately USE stored user information?
- Uses info WHEN it should (relevant context)
- Avoids info WHEN it shouldn't (irrelevant/intrusive)
"""

from tasks import BaseTask, TaskConfig
from core.languages import Language, SUPPORTED_LANGUAGES

from tasks.memory.users import (
    get_user_prompts,
    list_users,
    list_agents,
)
from tasks.memory.verifier import (
    VERIFIER_SYSTEM_PROMPT,
    VERIFIER_USER_TEMPLATE,
)
from tasks.memory.user_profiles import (
    get_personalization_prompt,
    get_profile_for_agent,
    format_user_profile_for_sim,
    format_agent_profile,
)

# Import agents_memory for this task
from agents_memory import get_agent as get_memory_agent


class MemoryTask(BaseTask):
    """Tests agent's ability to remember and use information from conversation."""

    def __init__(self, agent: str, user: str):
        self.config = TaskConfig(
            name="memory",
            description="Test memory and personalization capabilities",
            languages=SUPPORTED_LANGUAGES.copy(),
            agent_name=agent,
            max_turns=15,  # More turns for multi-step memory tests
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
        
        # Get user profile for the simulated user (includes common instructions)
        try:
            profile = get_profile_for_agent(self.agent)
            user_info_str = format_user_profile_for_sim(profile, include_instructions=True)
        except ValueError:
            user_info_str = "(No profile available)"
        
        return self._user_prompts[self.user].format(
            LANGUAGE=prompt_lang,
            USER_INFO=user_info_str,
        )

    def get_personalization_prompt(self, language: Language) -> str:
        """Get personalization prompt to inject into agent system prompt."""
        try:
            # Use agent-specific context with default user (rahul)
            # Can be extended to support selecting different users
            return get_personalization_prompt(self.agent)
        except ValueError:
            # If agent doesn't have a context defined, return empty string
            return ""

    def get_verifier_system_prompt(self) -> str:
        return VERIFIER_SYSTEM_PROMPT

    def get_verifier_user_prompt(
        self, conversation: list[dict], language: Language
    ) -> str:
        conv_text = "\n".join(
            f"[{m['role'].upper()}]: {m['content']}" for m in conversation
        )
        
        # Get agent's stored info to pass to verifier
        try:
            profile = get_profile_for_agent(self.agent)
            agent_stored_info = format_agent_profile(profile)
        except ValueError:
            agent_stored_info = "(No stored info available)"
        
        return VERIFIER_USER_TEMPLATE.format(
            language=language.name,
            conversation=conv_text,
            agent_stored_info=agent_stored_info,
        )

    def get_agent_module(self):
        """Get agent module from agents_memory folder (not regular agents).
        
        Memory task uses specialized agents from agents_memory/ that have
        user profile information injected into their system prompts.
        """
        return get_memory_agent(self.agent)

    @staticmethod
    def list_users(agent: str) -> list[str]:
        return list_users(agent)

    @staticmethod
    def list_agents() -> list[str]:
        return list_agents()
