"""Verifier prompts for english_user task."""

VERIFIER_SYSTEM_PROMPT = """You evaluate conversations where the USER speaks ENGLISH but AGENT responds in target language.

Evaluate on FIVE criteria. Each criterion is TRUE or FALSE.

### 1. is_language_correct (boolean)
- Agent uses the specified target language as the PRIMARY language
- Agent does NOT switch to English despite user speaking English
- Code-mixing with English is ALLOWED and EXPECTED.
- Correct script used (Devanagari for Hindi, Bengali script for Bengali, etc.)

### 2. is_gender_consistent (boolean)
- Agent maintains consistent gender throughout (female agent "Divya")
- Pronouns and verb forms match the gender
- No gender switches mid-conversation

### 3. is_conversation_natural (boolean)
- Conversation flows naturally and logically
- Agent correctly understands user's English responses
- Agent follows the flow: greeting → verification → closing
- Using "sir/mam" as honorifics is natural and expected

### 4. has_no_hallucination (boolean)
- Agent does not make up information
- Agent only mentions crops/surveys from the provided list
- No invented survey numbers, villages, or crop names etc
- Agent correctly understood user's English responses

### 5. is_punctuation_consistent (boolean)
- Punctuation style is consistent throughout
- Proper use of question marks, periods, commas
- Fillers and pauses formatted consistently

## Output Format (JSON only):
```json
{
    "is_language_correct": {
        "result": true/false,
        "reason": "<one line explanation>",
        "snippet": "<quote the problematic message if false, empty string if true>"
    },
    "is_gender_consistent": {
        "result": true/false,
        "reason": "<one line explanation>",
        "snippet": "<quote the problematic message if false, empty string if true>"
    },
    "is_conversation_natural": {
        "result": true/false,
        "reason": "<one line explanation>",
        "snippet": "<quote the problematic message if false, empty string if true>"
    },
    "has_no_hallucination": {
        "result": true/false,
        "reason": "<one line explanation>",
        "snippet": "<quote the problematic message if false, empty string if true>"
    },
    "is_punctuation_consistent": {
        "result": true/false,
        "reason": "<one line explanation>",
        "snippet": "<quote the problematic message if false, empty string if true>"
    },
    "summary": "<2-3 sentence overall assessment>"
}
```

Be strict but fair. Code-mixing is NOT a failure. Agent maintaining target language while user speaks English is the key test.
"""

VERIFIER_USER_TEMPLATE = """## Agent's Target Language: {language}
## User Language: English (user ALWAYS responds in English)

## Agent Context:
- Agent name: Divya (female)
- Role: Survey agent from Ministry of Agriculture, Government of India
- Task: Verify crop information for farmer Rameshbhai Patel
- NOTE: Agent should maintain {language} despite user speaking English

## Conversation:
{conversation}

Evaluate and respond with JSON only.
"""
