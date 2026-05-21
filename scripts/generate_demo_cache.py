import json
from pathlib import Path

from config import Config
from main import (
    SESSION_HISTORIES,
    RecommendRequest,
    SimulateRequest,
    recommend,
    simulate_review,
)


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "demo_cache"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def dump_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_pair(filename: str, endpoint: str, request: dict, response: dict) -> None:
    dump_json(
        OUTPUT_DIR / filename,
        {
            "endpoint": endpoint,
            "request": request,
            "response": response,
        },
    )


def main() -> None:
    Config.DEMO_MODE = False
    SESSION_HISTORIES.clear()

    t_tobi = {
        "name": "Tobi",
        "age": 27,
        "location": "Yaba, Lagos",
        "occupation": "freelance developer",
        "budget": "careful with money",
        "traits": ["budget-conscious", "likes practical gadgets", "hates anything without USB-C"],
    }

    amaka = {
        "name": "Amaka",
        "age": 31,
        "location": "Ikeja, Lagos",
        "occupation": "product manager",
        "budget": "mid-range",
        "traits": ["health-conscious", "likes dependable places", "prefers practical recommendations"],
    }

    # Scenario 1: review simulator backup for offline demo safety
    simulate_req = SimulateRequest(persona_description=t_tobi, product_id="B000BRLXUI")
    simulate_res = simulate_review(simulate_req)
    save_pair(
        "simulate_review_backup.json",
        "/simulate",
        simulate_req.model_dump(),
        simulate_res.model_dump(),
    )

    # Scenario 2: remote-work product recommendation
    recommend_req_1 = RecommendRequest(
        persona_description=t_tobi,
        message="I need something to help me work during blackouts",
        session_id=None,
    )
    recommend_res_1 = recommend(recommend_req_1)
    save_pair(
        "scenario1_recommend_remote_work.json",
        "/recommend",
        recommend_req_1.model_dump(),
        recommend_res_1.model_dump(),
    )

    # Scenario 3: follow-up in the same session that should bridge to a venue
    recommend_req_2 = RecommendRequest(
        persona_description=t_tobi,
        message="Yeah, find me a spot to use it",
        session_id=recommend_res_1.session_id,
    )
    recommend_res_2 = recommend(recommend_req_2)
    save_pair(
        "scenario2_recommend_bridge_spot.json",
        "/recommend",
        recommend_req_2.model_dump(),
        recommend_res_2.model_dump(),
    )

    # Scenario 4: different persona, different bridge (fitness)
    recommend_req_3 = RecommendRequest(
        persona_description=amaka,
        message="I want to get back into fitness and need a gym or studio near Ikeja with power and WiFi",
        session_id=None,
    )
    recommend_res_3 = recommend(recommend_req_3)
    save_pair(
        "scenario3_recommend_fitness_bridge.json",
        "/recommend",
        recommend_req_3.model_dump(),
        recommend_res_3.model_dump(),
    )

    print("Saved demo cache files:")
    for name in [
        "simulate_review_backup.json",
        "scenario1_recommend_remote_work.json",
        "scenario2_recommend_bridge_spot.json",
        "scenario3_recommend_fitness_bridge.json",
    ]:
        print(" -", OUTPUT_DIR / name)


if __name__ == "__main__":
    main()
