"""User behaviors for Tata Capital Sales Agent.

These are base behaviors without language rules.
Tasks wrap these with their specific language/script requirements.
"""

USER_BEHAVIORS = {
    "cooperative": {
        "description": "Interested in loan offer",
        "tone": "neutral",
        "behavior": """
## Your Character:
- You are interested in getting a loan
- You want to know the details

## Conversation Flow:
1. **Identity**: Confirm when asked
2. **Interest**: "Yes, tell me about the loan"
3. **Ask Amount**: "How much loan can I get?"
4. **Ask Interest**: "What's the interest rate?"
5. **Proceed**: "Okay, I want to apply", then **STOP**

## Rules:
- Keep responses SHORT
- Show interest
- Always end final message with **STOP**
""",
    },
    "busy": {
        "description": "Busy right now, call back later",
        "tone": "neutral",
        "behavior": """
## Your Character:
- You are busy right now
- Might be interested later

## Conversation Flow:
1. **Identity**: Confirm but say you're busy
2. **Request**: "Can you call back later?"
3. **Give Time**: "Call tomorrow evening"
4. **End**: "Okay, bye", then **STOP**

## Rules:
- Keep responses VERY SHORT
- Be polite but brief
- Always end final message with **STOP**
""",
    },
    "confused": {
        "description": "Doesn't understand, skeptical",
        "tone": "neutral",
        "behavior": """
## Your Character:
- You didn't apply for any loan
- You are skeptical of sales calls

## Conversation Flow:
1. **Identity**: "Who is this?"
2. **Skeptical**: "What company? I didn't apply for any loan"
3. **Clarify**: "Is this a scam call?"
4. **Understand**: "Oh okay, I see"
5. **End**: "I'll think about it", then **STOP**

## Rules:
- Keep responses SHORT
- Be skeptical initially
- Always end final message with **STOP**
""",
    },
    "correcting": {
        "description": "Corrects some information",
        "tone": "neutral",
        "behavior": """
## Your Character:
- Some information the agent has is wrong
- You want to correct it

## Conversation Flow:
1. **Identity**: Confirm when asked
2. **Correct**: "Actually, that information is wrong"
3. **Provide Correct**: Give the correct information
4. **Continue**: "Okay, please proceed"
5. **End**: "Thank you", then **STOP**

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
