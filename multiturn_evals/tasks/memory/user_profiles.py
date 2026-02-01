"""Agent profiles for memory/personalization testing.

This module contains what the AGENT has stored about the user (like a CRM database).
This information is injected into the agent's system prompt.

The SIMULATED USER has NO knowledge of what the agent has stored.
User behavior/steps are defined separately in the users/ folder.

This separation allows testing:
- Does the agent correctly USE the stored information?
- Does the agent ask redundant questions about info it already has?
- Does the agent personalize based on business context?
"""

# =============================================================================
# USER PROFILES - Base user data (personal info, preferences)
# =============================================================================
# Add new users here. Each user can then be combined with any agent context.
# Profiles include domain-relevant info for different agent types:
# - Financial: income, employment, bank details
# - Hiring: skills, experience, availability
# - Farming: land, crops, schemes
# - General: food preferences, lifestyle

USER_RAHUL = {
    "personal_info": [
        "Name: Rahul Kumar",
        "Location: Mumbai, Maharashtra (Andheri West)",
        "Age: 28",
        "Profession: Software Engineer at TCS",
        "Phone: 98765-43210",
    ],
    # Financial info (for loan agents)
    "financial_info": [
        "Monthly salary: ₹75,000",
        "Salary credit date: 1st of every month",
        "Bank: HDFC Bank, Andheri branch",
        "Account ending: ***4521",
    ],
    # Work info (for hiring agents)
    "work_info": [
        "Current employer: TCS (3 years)",
        "Skills: Python, JavaScript, React",
        "Availability: Weekends only",
        "Work preference: Remote or hybrid",
    ],
    "preferences": [
        "Prefers spicy Indian food, especially street food",
        "Enjoys reading in peaceful, quiet places",
        "Likes visiting bookstores and cafes",
        "Prefers evening calls (after 6 PM)",
    ],
    "dislikes": [
        "Very crowded places",
        "Noisy environments",
        "Morning calls before 9 AM",
    ],
}

USER_PRIYA = {
    "personal_info": [
        "Name: Priya Sharma",
        "Location: Delhi, NCR (Dwarka Sector 12)",
        "Age: 32",
        "Profession: Marketing Manager at Flipkart",
        "Phone: 99887-76655",
    ],
    "financial_info": [
        "Monthly salary: ₹1,20,000",
        "Salary credit date: Last working day of month",
        "Bank: ICICI Bank, Dwarka branch",
        "Account ending: ***8832",
    ],
    "work_info": [
        "Current employer: Flipkart (5 years)",
        "Skills: Digital marketing, Team management",
        "Availability: Weekdays after 7 PM",
        "Work preference: Office-based",
    ],
    "preferences": [
        "Loves South Indian food, especially dosa",
        "Enjoys yoga and meditation",
        "Prefers online shopping over malls",
        "Prefers WhatsApp communication over calls",
    ],
    "dislikes": [
        "Late deliveries",
        "Aggressive sales pitches",
        "Repeated follow-up calls",
    ],
}

USER_RAMESH = {
    "personal_info": [
        "Name: Ramesh Patel",
        "Location: Ahmedabad, Gujarat (Satellite area)",
        "Age: 45",
        "Profession: Small Business Owner - Electronics Shop",
        "Phone: 94265-12345",
    ],
    "financial_info": [
        "Monthly business income: ₹2-3 lakhs (variable)",
        "Business account: Bank of Baroda",
        "Account ending: ***7721",
        "GST registered: Yes",
    ],
    "work_info": [
        "Business: Patel Electronics (15 years)",
        "Shop location: CG Road, Ahmedabad",
        "Staff: 3 employees",
        "Shop timings: 10 AM - 9 PM",
    ],
    # Farming info (for DCS - Ramesh also has ancestral farmland)
    "farming_info": [
        "Ancestral land: 5 hectares in Banas Kantha",
        "Crops: Groundnut, Cotton (leased to relatives)",
        "PM-KISAN: Not registered (managed by relatives)",
    ],
    "preferences": [
        "Prefers Gujarati food",
        "Values family time",
        "Likes straightforward conversations",
        "Prefers calls in Gujarati or Hindi",
    ],
    "dislikes": [
        "Complicated processes",
        "Long waiting times",
        "Too much paperwork",
    ],
}

USER_ANITA = {
    "personal_info": [
        "Name: Anita Devi",
        "Location: Patna, Bihar (Boring Road)",
        "Age: 38",
        "Profession: School Teacher at DPS Patna",
        "Phone: 78903-45678",
    ],
    "financial_info": [
        "Monthly salary: ₹45,000",
        "Salary credit date: 5th of every month",
        "Bank: State Bank of India",
        "Account ending: ***1199",
    ],
    "work_info": [
        "Current employer: DPS Patna (10 years)",
        "Subjects: Mathematics, Science",
        "Availability: After school hours (3 PM onwards)",
        "Summer vacation: May-June",
    ],
    "preferences": [
        "Prefers home-cooked food",
        "Enjoys reading and gardening",
        "Values education and learning",
        "Prefers Hindi communication",
    ],
    "dislikes": [
        "Rude behavior",
        "Unnecessary formality",
        "Calls during school hours",
    ],
}

# Registry of user profiles - add new users here
USER_PROFILES = {
    "rahul": USER_RAHUL,
    "priya": USER_PRIYA,
    "ramesh": USER_RAMESH,
    "anita": USER_ANITA,
}

# Default user
DEFAULT_USER = "rahul"

# =============================================================================
# Agent-specific business context and previous interactions
# =============================================================================

# IDFC First Bank - EMI Collection Agent
AGENT_CONTEXT_IDFC = {
    "business_context": "Personal loan of ₹2,00,000 taken 8 months ago. EMI: ₹5,000/month. Never missed a single EMI until last month. Last month's EMI bounced due to insufficient balance. Total outstanding: ₹5,000 + ₹500 late fee.",
    "previous_interactions": [
        "Called 5 days ago - user said salary was delayed, promised to pay by weekend",
        "SMS reminder sent 3 days ago",
        "User has been a good customer with no previous defaults",
    ],
}

# Tata Capital - Loan Sales Agent
AGENT_CONTEXT_TATA_CAP = {
    "business_context": "Existing Tata Capital customer. Has a running business loan of ₹5,00,000 taken 2 years ago, repaying on time. Pre-approved for new loan of ₹8,50,000 based on good repayment history. Business type: Retail electronics shop.",
    "previous_interactions": [
        "Received promotional SMS about pre-approved loan offer last week",
        "Customer enquired about loan top-up 3 months ago but didn't proceed",
        "Has been with Tata Capital for 2 years with perfect repayment record",
    ],
}

# Urban Company - Scheduling/Hiring Agent
AGENT_CONTEXT_UC = {
    "business_context": "Applied for electrician partner role on Urban Company website 3 days ago. Currently works as freelance electrician in Hyderabad. Has 5 years of experience. Lives in Kukatpally area.",
    "previous_interactions": [
        "Submitted application form online with basic details",
        "Received confirmation SMS that UC team will call",
        "First time applicant to Urban Company",
    ],
}

# DCS - Farmer Crop Verification Survey Agent
AGENT_CONTEXT_DCS = {
    "business_context": "Registered farmer under PM-KISAN scheme. Has 4 survey plots in Banas Kantha district, Gujarat. Primary crops: Groundnut (Magfali), Cotton (Kapas). Total land holding: 8 hectares.",
    "previous_interactions": [
        "Participated in last year's crop survey - all details verified correctly",
        "Received PM-KISAN benefit successfully for last 3 installments",
        "No disputes or discrepancies in previous land records",
    ],
}

# General Assistant
AGENT_CONTEXT_GENERAL_ASSISTANT = {
    "business_context": "",  # General assistant doesn't need specific business context
    "previous_interactions": [
        "Asked about weather in Mumbai last week",
        "Enquired about nearby restaurants 2 days ago",
        "Helped with translation from Hindi to English yesterday",
    ],
}


# =============================================================================
# COMMON INSTRUCTIONS for simulated users
# =============================================================================
# These instructions are injected into ALL user prompts

USER_PROFILE_INSTRUCTIONS = """
## IMPORTANT - How to use your profile:
- You KNOW this information about yourself
- But DO NOT volunteer it upfront - let the agent demonstrate they know it
- Only reveal your info when the agent specifically asks for it
- Even when asked, only reveal the specific information requested, not everything
- Do not reveal this info unless explicitly asked.
""".strip()


# Format user profile for simulated user (what user knows about themselves)
def format_user_profile_for_sim(
    profile: dict, include_instructions: bool = True
) -> str:
    """Format user profile for the simulated user's prompt.

    This is what the simulated user knows about themselves.
    Does NOT include business_context or previous_interactions (user doesn't know those).

    Args:
        profile: User profile dictionary
        include_instructions: Whether to include common instructions (default True)
    """
    lines = []

    lines.append("## WHO YOU ARE:")

    # Personal Info
    if profile.get("personal_info"):
        for info in profile["personal_info"]:
            lines.append(f"- {info}")

    # Financial Info (user knows their own finances)
    if profile.get("financial_info"):
        lines.append("")
        lines.append("**Your Financial Info:**")
        for info in profile["financial_info"]:
            lines.append(f"- {info}")

    # Work Info (user knows their work details)
    if profile.get("work_info"):
        lines.append("")
        lines.append("**Your Work Info:**")
        for info in profile["work_info"]:
            lines.append(f"- {info}")

    # Farming Info (user knows their farming details)
    if profile.get("farming_info"):
        lines.append("")
        lines.append("**Your Farming Info:**")
        for info in profile["farming_info"]:
            lines.append(f"- {info}")

    # Preferences
    if profile.get("preferences"):
        lines.append("")
        lines.append("**Your Preferences:**")
        for preference in profile["preferences"]:
            lines.append(f"- {preference}")

    # Dislikes
    if profile.get("dislikes"):
        lines.append("")
        lines.append("**Things you dislike:**")
        for dislike in profile["dislikes"]:
            lines.append(f"- {dislike}")

    # Add common instructions
    if include_instructions:
        lines.append("")
        lines.append(USER_PROFILE_INSTRUCTIONS)

    return "\n".join(lines)


# Format agent profile as a string to inject into agent's system prompt
def format_agent_profile(profile: dict) -> str:
    """Format agent profile (what agent knows) for injection into agent's system prompt."""
    lines = [
        "## USER CONTEXT (Remember this information throughout the conversation):",
        "",
    ]

    # Personal Info
    if profile.get("personal_info"):
        lines.append("**Personal Information:**")
        for info in profile["personal_info"]:
            lines.append(f"- {info}")
        lines.append("")

    # Financial Info (for loan/banking agents)
    if profile.get("financial_info"):
        lines.append("**Financial Information:**")
        for info in profile["financial_info"]:
            lines.append(f"- {info}")
        lines.append("")

    # Work Info (for hiring agents)
    if profile.get("work_info"):
        lines.append("**Work/Employment Information:**")
        for info in profile["work_info"]:
            lines.append(f"- {info}")
        lines.append("")

    # Farming Info (for agriculture/survey agents)
    if profile.get("farming_info"):
        lines.append("**Farming Information:**")
        for info in profile["farming_info"]:
            lines.append(f"- {info}")
        lines.append("")

    # Preferences
    if profile.get("preferences"):
        lines.append("**Preferences:**")
        for preference in profile["preferences"]:
            lines.append(f"- {preference}")
        lines.append("")

    # Dislikes
    if profile.get("dislikes"):
        lines.append("**Dislikes:**")
        for dislike in profile["dislikes"]:
            lines.append(f"- {dislike}")
        lines.append("")

    # Business Context (agent-specific)
    if profile.get("business_context"):
        lines.append("**Business Context (Agent-Specific):**")
        lines.append(f"- {profile['business_context']}")
        lines.append("")

    # Previous Interactions (agent-specific)
    if profile.get("previous_interactions"):
        lines.append("**Previous Interactions:**")
        for interaction in profile["previous_interactions"]:
            lines.append(f"- {interaction}")
        lines.append("")

    lines.extend(
        [
            "**Important:** Use this information to personalize your responses.",
            "Remember the user's preferences and context throughout the conversation.",
            "Don't ask for information that's already provided above.",
            "You should customise your way of speaking to the user basis the user's information.",
        ]
    )

    return "\n".join(lines)


# Registry mapping agent names to their specific context
AGENT_CONTEXTS = {
    "idfc_main": AGENT_CONTEXT_IDFC,
    "tata_cap_sales": AGENT_CONTEXT_TATA_CAP,
    "uc_scheduling": AGENT_CONTEXT_UC,
    "dcs": AGENT_CONTEXT_DCS,
    "general_assistant": AGENT_CONTEXT_GENERAL_ASSISTANT,
}


def get_user_profile(user_name: str = DEFAULT_USER) -> dict:
    """Get user profile by name."""
    if user_name not in USER_PROFILES:
        available = ", ".join(USER_PROFILES.keys())
        raise ValueError(f"Unknown user: {user_name}. Available: {available}")
    return USER_PROFILES[user_name].copy()


def list_available_users() -> list[str]:
    """List all available user profiles."""
    return list(USER_PROFILES.keys())


def get_agent_context(agent_name: str) -> dict:
    """Get agent-specific context (business_context and previous_interactions)."""
    if agent_name not in AGENT_CONTEXTS:
        available = ", ".join(AGENT_CONTEXTS.keys())
        raise ValueError(f"Unknown agent: {agent_name}. Available: {available}")
    return AGENT_CONTEXTS[agent_name].copy()


def get_profile_for_agent(agent_name: str, user_name: str = DEFAULT_USER) -> dict:
    """Get user profile combined with agent-specific context.

    Args:
        agent_name: The agent name (e.g., 'idfc_main', 'tata_cap_sales')
        user_name: The user profile name (e.g., 'rahul', 'priya')

    Returns:
        Combined profile with user info + agent-specific context
    """
    profile = get_user_profile(user_name)
    agent_context = get_agent_context(agent_name)

    # Merge agent-specific context into profile
    profile["business_context"] = agent_context.get("business_context", "")
    profile["previous_interactions"] = agent_context.get("previous_interactions", [])

    return profile


def get_personalization_prompt(agent_name: str, user_name: str = DEFAULT_USER) -> str:
    """Get formatted personalization prompt for injection into agent's system prompt.

    Args:
        agent_name: The agent name (e.g., 'idfc_main', 'tata_cap_sales')
        user_name: The user profile name (e.g., 'rahul', 'priya')
    """
    profile = get_profile_for_agent(agent_name, user_name)
    return format_agent_profile(profile)
