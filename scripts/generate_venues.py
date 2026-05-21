import json
import random
import os

def generate_venues():
    cities = {
        "Lagos": ["Ikeja", "Victoria Island", "Lekki", "Surulere", "Yaba", "Maryland", "Ajah", "Apapa"],
        "Abuja": ["Maitama", "Wuse II", "Garki", "Asokoro", "Jabi", "Wuye", "Utako"]
    }

    buckets = [
        {"category": "Remote Work Cafe", "count": 40, "categories": ["Cafe", "Workspace", "Coffee Shop"]},
        {"category": "Fitness/Gym", "count": 30, "categories": ["Gym", "Fitness Center", "Yoga Studio"]},
        {"category": "Restaurant", "count": 40, "categories": ["Restaurant", "Dining", "Bistro"]},
        {"category": "Event Venue/Tailor", "count": 25, "categories": ["Event Space", "Tailor", "Fashion House"]},
        {"category": "Family/Clinic", "count": 25, "categories": ["Clinic", "Family Center", "Pediatrics"]}
    ]

    all_venues = []
    
    # Pre-defined naming patterns for variety
    names_map = {
        "Remote Work Cafe": ["The Hub", "Co-Work Lagos", "Bean & Byte", "Digital Nomad Cafe", "Quiet Corner", "Brew & Browse", "Cyber Sips", "Focus Point"],
        "Fitness/Gym": ["Iron Temple", "Peak Performance", "Zen Yoga", "Lagos Fit", "Abuja Athletics", "Muscle Mansion", "Core Strength", "Vitality Gym"],
        "Restaurant": ["Spice Route", "Jollof Junction", "The Grill House", "Taste of Naija", "Savory Bites", "Coastal Cuisines", "Urban Eats", "Heritage Kitchen"],
        "Event Venue/Tailor": ["Royal Events", "Stitch & Style", "Elite Occasions", "Golden Needle", "Vogue Atelier", "Grand Ballroom", "Precision Tailoring", "Celebration Hall"],
        "Family/Clinic": ["Care First Clinic", "Happy Hearts Pediatric", "Family Wellness", "City Health Center", "Gentle Care", "Bloom Clinic", "Parent & Child", "Safe Haven Health"]
    }

    for bucket in buckets:
        for i in range(bucket["count"]):
            city = random.choice(list(cities.keys()))
            area = random.choice(cities[city])
            
            name_base = random.choice(names_map[bucket["category"]])
            name = f"{name_base} {random.choice(['Center', 'Hub', 'Place', 'Spot', 'Express', ''])}{' ' + area if random.random() > 0.5 else ''}".strip()
            
            # Random coordinates around Nigeria
            if city == "Lagos":
                lat = random.uniform(6.4, 6.6)
                lon = random.uniform(3.3, 3.5)
            else:
                lat = random.uniform(9.0, 9.1)
                lon = random.uniform(7.4, 7.5)

            venue = {
                "name": name,
                "address": f"{random.randint(1, 500)} {random.choice(['Street', 'Road', 'Way', 'Avenue', 'Close'])}",
                "city": city,
                "area": area,
                "latitude": round(lat, 4),
                "longitude": round(lon, 4),
                "stars": round(random.uniform(3.0, 5.0), 1),
                "review_count": random.randint(10, 500),
                "categories": bucket["categories"] + [random.choice(["Popular", "Highly Rated", "New", "Local Favorite"])],
                "hours": f"{random.randint(7, 10)}:00 AM - {random.randint(18, 23)}:00 {random.choice(['PM', 'AM'])}",
                "has_generator": False, # Assigned later
                "has_wifi": random.choice([True, False]),
                "open_late": random.choice([True, False]),
                "family_friendly": random.choice([True, False]),
                "parking": random.choice([True, False])
            }
            all_venues.append(venue)

    # Precisely set ~40 generators
    generator_indices = random.sample(range(len(all_venues)), 40)
    for idx in generator_indices:
        all_venues[idx]["has_generator"] = True

    # Ensure specific logic for categories (wifi for cafes)
    for v in all_venues:
        if "Cafe" in v["categories"] or "Workspace" in v["categories"]:
            v["has_wifi"] = True

    # Write to JSONL
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/nigerian_venues.jsonl", "w", encoding="utf-8") as f:
        for venue in all_venues:
            f.write(json.dumps(venue) + "\n")

    print(f"Generated {len(all_venues)} venues in data/processed/nigerian_venues.jsonl")
    print(f"Generator count: {sum(1 for v in all_venues if v['has_generator'])}")

if __name__ == "__main__":
    generate_venues()