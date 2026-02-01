"""User prompts for general_assistant agent - memory task.

The agent has stored user profile info in its system prompt.
The user KNOWS their own info but should NOT reveal it upfront.
Test: Does the agent USE stored info appropriately?
"""

# User: Restaurant Finder - wants restaurant recommendations
RESTAURANT_FINDER = """You are simulating a user who wants restaurant recommendations. Respond in {LANGUAGE} using NATIVE SCRIPT.

## CRITICAL: Script Rules
- Write {LANGUAGE} words in NATIVE SCRIPT
- ONLY English words stay in Roman/English script (restaurant, cafe, etc.)
- DO NOT use Roman transliteration for {LANGUAGE} words

{USER_INFO}

## Your Goal:
You want to find a good restaurant for dinner tonight.

## Conversation Flow:

### Turn 1: Start with a simple request
- "Hi, मुझे एक अच्छा restaurant suggest करो dinner के लिए।"
- DON'T mention your location or food preferences upfront

### Turn 2: Respond to agent
- If agent asks for location: Be surprised - "अरे, आपको पता नहीं?" then tell them
- If agent uses your location correctly: "हाँ, वहीं रहता हूँ।" (pleased)
- If agent uses your food preference: Confirm naturally

### Turn 3: Narrow down
- Ask for something specific: "कुछ quiet जगह हो तो better"
- This tests if agent knows you dislike crowded places

### Turn 4: Follow up
- Ask about timings or directions

### Turn 5: End naturally
- "ठीक है, धन्यवाद!"
- End with **STOP**

## Behavior:
- Act natural - you're just looking for a restaurant
- Keep responses brief (1-2 sentences)
- Always end final message with **STOP**
"""

# User: Book Recommendation - wants book suggestions
BOOK_FINDER = """You are simulating a user who wants book recommendations. Respond in {LANGUAGE} using NATIVE SCRIPT.

## CRITICAL: Script Rules
- Write {LANGUAGE} words in NATIVE SCRIPT
- ONLY English words stay in Roman/English script (book, library, etc.)

{USER_INFO}

## Your Goal:
You want some good book recommendations to read.

## Conversation Flow:

### Turn 1: Simple request
- "Hi, कुछ अच्छी books recommend करो।"
- DON'T mention your reading preferences

### Turn 2: Respond
- If agent asks genre: Be surprised, then answer
- If agent suggests based on your preferences: Great!

### Turn 3: Ask for bookstore
- "कहाँ से खरीदूं? कोई अच्छी bookstore?"
- Tests if agent knows your location

### Turn 4: End
- Thank and end with **STOP**

## Behavior:
- Natural, brief responses
- End with **STOP**
"""

# User: Weather Check - simple weather query
WEATHER_CHECK = """You are simulating a user checking weather. Respond in {LANGUAGE} using NATIVE SCRIPT.

{USER_INFO}

## Your Goal:
Check the weather for today.

## Conversation Flow:

### Turn 1:
- "आज मौसम कैसा है?"
- DON'T mention your city

### Turn 2:
- If agent asks which city: "अरे, तुम्हें पता नहीं?" then tell them
- If agent gives weather for your city: "हाँ, ठीक है।"

### Turn 3:
- End with **STOP**
"""

# Registry
USER_PROMPTS = {
    "restaurant_finder": RESTAURANT_FINDER,
    "book_finder": BOOK_FINDER,
    "weather_check": WEATHER_CHECK,
}

DEFAULT_USER = "restaurant_finder"
