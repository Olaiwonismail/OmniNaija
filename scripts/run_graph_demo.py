from agent.graph import understand_user, retrieve_products, should_bridge, retrieve_locations, compose_response
import json

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
    print("\n=== Demo run for message:")
    print(msg)
    state = understand_user(msg, persona, chat_history=[{"role": "user", "content": msg}])

    # Retrieve candidate products using the user's message as query
    products = retrieve_products(msg, top_k=5)
    state['products'] = products

    if should_bridge(state):
        bridge_cat = state['intent'].get('bridge_category')
        locations = retrieve_locations(bridge_category=bridge_cat, top_k=3)
        state['locations'] = locations
    else:
        state['locations'] = []

    try:
        output = compose_response(state, top_k_products=3)
    except Exception as e:
        output = f"Error composing response: {e}"

    print("\n--- Final recommendation ---")
    print(output)
    print("\n--- Debug state ---")
    print(json.dumps({
        'intent': state.get('intent'),
        'products': [p.get('product_id') for p in products[:3]],
        'bridged': should_bridge(state),
        'locations': [l.get('venue_id') for l in state.get('locations', [])]
    }, indent=2, ensure_ascii=False))
