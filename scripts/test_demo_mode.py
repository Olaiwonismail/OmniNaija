import json
from pathlib import Path

from config import Config
from main import SESSION_HISTORIES, RecommendRequest, SimulateRequest, recommend, simulate_review


CACHE_DIR = Path(__file__).resolve().parent.parent / "demo_cache"


def load_case(name: str) -> dict:
    return json.loads((CACHE_DIR / name).read_text(encoding="utf-8"))


def assert_equal(label: str, got: dict, expected: dict) -> None:
    if got != expected:
        raise AssertionError(f"{label} mismatch\nGOT:\n{json.dumps(got, ensure_ascii=False, indent=2)}\nEXPECTED:\n{json.dumps(expected, ensure_ascii=False, indent=2)}")
    print(f"PASS: {label}")


def main() -> None:
    Config.DEMO_MODE = True
    SESSION_HISTORIES.clear()

    simulate_case = load_case("simulate_review_backup.json")
    scenario1 = load_case("scenario1_recommend_remote_work.json")
    scenario2 = load_case("scenario2_recommend_bridge_spot.json")
    scenario3 = load_case("scenario3_recommend_owambe_bridge.json")

    sim_req = SimulateRequest(**simulate_case["request"])
    sim_res = simulate_review(sim_req)
    assert_equal("simulate_review_backup", sim_res.model_dump(), simulate_case["response"])

    req1 = RecommendRequest(**scenario1["request"])
    res1 = recommend(req1)
    assert_equal("scenario1_recommend_remote_work", res1.model_dump(), scenario1["response"])

    req2 = RecommendRequest(**scenario2["request"])
    res2 = recommend(req2)
    assert_equal("scenario2_recommend_bridge_spot", res2.model_dump(), scenario2["response"])

    req3 = RecommendRequest(**scenario3["request"])
    res3 = recommend(req3)
    assert_equal("scenario3_recommend_owambe_bridge", res3.model_dump(), scenario3["response"])

    print("\nAll demo mode cache checks passed.")


if __name__ == "__main__":
    main()
