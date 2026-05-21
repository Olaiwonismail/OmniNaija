from main import recommend, RecommendRequest, SESSION_HISTORIES

persona = {"name": "Tobi", "age": 27, "location": "Yaba, Lagos", "occupation": "freelance developer", "budget": "careful with money", "traits": ["budget-conscious"]}

# First message: expect product recommendation
req1 = RecommendRequest(persona_description=persona, message="I need something to help me work during blackouts", session_id=None)
res1 = recommend(req1)
print("Session ID:", res1.session_id)
print("Recommendation snippet:\n", res1.recommendation[:400])
print("History length:", len(res1.history))

# Second message: follow-up asking to find a spot — should bridge
sid = res1.session_id
req2 = RecommendRequest(persona_description=persona, message="Yeah, find me a spot to use it", session_id=sid)
res2 = recommend(req2)
print("\nSecond turn recommendation snippet:\n", res2.recommendation[:400])
print("History length after second turn:", len(res2.history))
print("Last two history entries:", res2.history[-2:])
