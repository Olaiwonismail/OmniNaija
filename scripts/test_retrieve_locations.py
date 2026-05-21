from agent.graph import should_bridge, retrieve_locations

# Example states
state1 = {"intent": {"intent": "remote_work_setup", "confidence": 0.95, "bridge_category": "cafes_with_power"}}
state2 = {"intent": {"intent": "general_browsing", "confidence": 0.45, "bridge_category": None}}

print("STATE1 should bridge:", should_bridge(state1))
print("STATE2 should bridge:", should_bridge(state2))

print("\nTop venues for STATE1 bridge category:")
venues = retrieve_locations(bridge_category=state1['intent']['bridge_category'], top_k=3)
for v in venues:
    print(f"- {v.get('venue_id')} | {v.get('name')} | gen:{v.get('has_generator')} | wifi:{v.get('has_wifi')}")

print("\nGeneric top venues:")
venues2 = retrieve_locations(top_k=3)
for v in venues2:
    print(f"- {v.get('venue_id')} | {v.get('name')} | gen:{v.get('has_generator')} | wifi:{v.get('has_wifi')}")
