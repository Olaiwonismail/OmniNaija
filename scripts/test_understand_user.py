from agent.graph import understand_user

persona = {
    "name": "Tobi",
    "age": 27,
    "location": "Yaba, Lagos",
    "occupation": "freelance developer",
    "budget": "careful with money",
    "traits": ["budget-conscious", "likes practical gadgets", "hates anything without USB-C"],
}

examples = [
    "I need something to help me work during blackouts",
    "just looking for a phone case",
]

for msg in examples:
    state = {}
    out = understand_user(msg, persona, chat_history=[{"role": "user", "content": msg}], state=state)
    print("MESSAGE:", msg)
    print(out)
    print()
