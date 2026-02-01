"""User prompts for Urban Company Scheduling Agent - memory task.

The agent has stored info (application details, experience, location).
The user knows their own info but should NOT volunteer it upfront.
Test: Does the agent USE stored info appropriately?
"""

# User: Interested - wants to proceed with application
INTERESTED = """You are simulating a user receiving a hiring call. Respond in {LANGUAGE} using NATIVE SCRIPT.

## CRITICAL: Script Rules
- Write {LANGUAGE} words in NATIVE SCRIPT
- ONLY English words stay in Roman/English script (Urban Company, interview, etc.)

{USER_INFO}

## Your Goal:
You're receiving a call about a job application. You want to know more and possibly schedule an interview.

## Conversation Flow:

### Turn 1: Confirm Identity
- "हाँ, बोल रहा हूँ"

### Turn 2: Show Interest
- "हाँ, interested हूँ। बताइए"
- TEST: Does agent know you applied?

### Turn 3: Ask About Earnings
- "कितना earn कर सकते हैं?"

### Turn 4: Ask About Office
- "Interview कहाँ होगा?"
- TEST: Does agent suggest office near your area?

### Turn 5: Schedule
- "ठीक है, आ जाऊंगा"
- **STOP**

## Behavior:
- Interested and cooperative
- Brief responses
- End with **STOP**
"""

# User: Has Questions - wants clarity before committing
HAS_QUESTIONS = """You are simulating a user with questions. Respond in {LANGUAGE} using NATIVE SCRIPT.

{USER_INFO}

## Your Goal:
Understand the opportunity better before committing.

## Conversation Flow:

### Turn 1: Confirm
- "हाँ, बोल रहा हूँ"

### Turn 2: Ask How It Works
- "काम कैसे मिलता है?"

### Turn 3: Ask About Flexibility
- "अपने time पर काम कर सकते हैं?"

### Turn 4: Decide
- "सोचकर बताता हूँ"
- **STOP**
"""

# Registry
USER_PROMPTS = {
    "interested": INTERESTED,
    "has_questions": HAS_QUESTIONS,
}

DEFAULT_USER = "interested"
