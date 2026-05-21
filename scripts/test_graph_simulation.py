from main import run_graph_simulation, parse_persona

persona = {
    "name": "Tobi",
    "age": 27,
    "location": "Yaba, Lagos",
    "occupation": "freelance developer",
    "budget": "careful with money",
    "traits": ["budget-conscious", "likes practical gadgets", "hates anything without USB-C"],
}

message = "I need something to help me work during blackouts"

persona_parsed = parse_persona(persona)
res = run_graph_simulation(persona_parsed, message)
print("--- Recommendation ---")
print(res["recommendation"][:1000])
print("\n--- Debug ---")
print(res["state"]["intent"])
print([p.get("product_id") for p in res["state"]["products"][:3]])
print("Bridged:", res["state"]["bridged"])
