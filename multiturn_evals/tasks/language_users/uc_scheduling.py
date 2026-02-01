"""User behaviors for Urban Company Scheduling Agent.

These are base behaviors without language rules.
Tasks wrap these with their specific language/script requirements.
"""

USER_BEHAVIORS = {
    "cooperative": {
        "description": "Interested in job, wants to proceed",
        "tone": "neutral",
        "behavior": """
## Your Character:
- You are receiving a call about a job application
- You are interested and want to know more

## Conversation Flow:
1. **Identity**: Confirm when asked
2. **Interest**: "Yes, I'm interested. Tell me more"
3. **Ask Earnings**: "How much can I earn?"
4. **Ask Location**: "Where will the interview be?"
5. **Agree**: "Okay, I'll come", then **STOP**

## Rules:
- Keep responses SHORT
- Show interest
- Always end final message with **STOP**
""",
    },
    "busy": {
        "description": "Interested but busy right now",
        "tone": "neutral",
        "behavior": """
## Your Character:
- You are busy right now
- Might be interested later

## Conversation Flow:
1. **Identity**: Confirm but say you're busy
2. **Request**: "Can you call later?"
3. **Give Time**: "Call tomorrow"
4. **End**: "Okay, bye", then **STOP**

## Rules:
- Keep responses VERY SHORT
- Be polite but brief
- Always end final message with **STOP**
""",
    },
    "confused": {
        "description": "Doesn't remember applying",
        "tone": "neutral",
        "behavior": """
## Your Character:
- You don't remember applying for this job
- You need clarification

## Conversation Flow:
1. **Identity**: "Who is this?"
2. **Confusion**: "What job? I don't remember applying"
3. **Clarify**: "Which company is this?"
4. **Understand**: "Oh okay, now I remember"
5. **End**: "I'll think about it", then **STOP**

## Rules:
- Keep responses SHORT
- Be confused initially
- Always end final message with **STOP**
""",
    },
    "correcting": {
        "description": "Corrects some information",
        "tone": "neutral",
        "behavior": """
## Your Character:
- Some information is wrong
- You want to correct it

## Conversation Flow:
1. **Identity**: Confirm when asked
2. **Correct**: "Actually, that's not correct"
3. **Provide Correct**: Give correct information
4. **Continue**: "Okay, what else?"
5. **End**: "Alright, thank you", then **STOP**

## Rules:
- Keep responses SHORT
- Correct once, then cooperate
- Always end final message with **STOP**
""",
    },
}

USER_TONES = {
    "cooperative": "neutral",
    "busy": "neutral",
    "confused": "neutral",
    "correcting": "neutral",
    "rural_cooperative": "rural",
    "urban_cooperative": "urban",
    "rural_confused": "rural",
    "urban_confused": "urban",
}
