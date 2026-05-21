import json
import os
from pathlib import Path
from typing import Any

import chromadb
import requests
import streamlit as st


APP_TITLE = "OmniNaija Intent Graph"
DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
PERSONAS_PATH = Path(__file__).resolve().parent.parent / "personas" / "personas.json"
CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
CHROMA_COLLECTION = "amazon_products"

REVIEW_PICKER_IDS = [
    "Electronics:B07Y11LT52",
    "Electronics:B0C1FRBK4K",
    "Electronics:B0BB6TYK7V",
    "Electronics:B07S5JVP78",
    "Home_and_Kitchen:B000BRLXUI",
    "Home_and_Kitchen:B0917BW1RH",
    "Home_and_Kitchen:B0B2CMJ8TQ",
    "Home_and_Kitchen:B0BH7HRYZV",
    "Books:0307744434",
    "Books:B00HOV4GEO",
    "Books:B002SXIF4A",
    "Books:B008BU74RS",
]


@st.cache_data
def load_personas() -> list[dict[str, Any]]:
    return json.loads(PERSONAS_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_review_products(limit: int = 12) -> list[dict[str, Any]]:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name=CHROMA_COLLECTION)
    result = collection.get(ids=REVIEW_PICKER_IDS[:limit], include=["metadatas", "documents"])

    products: list[dict[str, Any]] = []
    for idx, product_id in enumerate(result.get("ids", [])):
        metadata = dict(result.get("metadatas", [{}])[idx] or {})
        products.append(
            {
                "product_id": product_id,
                "title": metadata.get("title") or product_id,
                "category": metadata.get("category") or "",
                "brand": metadata.get("brand") or "",
                "price": metadata.get("price"),
            }
        )
    return products


def persona_card(persona_entry: dict[str, Any]) -> str:
    persona = persona_entry["persona"]
    traits = ", ".join(persona.get("traits", []))
    return (
        f"**{persona_entry['avatar']}  {persona_entry['name']}**  \n"
        f"{persona_entry['bio']}  \n"
        f"_Location:_ {persona.get('location', 'Unknown')}  \n"
        f"_Traits:_ {traits}"
    )


def init_state() -> None:
    personas = load_personas()
    products = load_review_products()
    if "selected_persona_id" not in st.session_state:
        st.session_state.selected_persona_id = personas[0]["id"]
    if "selected_persona" not in st.session_state:
        st.session_state.selected_persona = personas[0]["persona"]
    if "selected_product_id" not in st.session_state and products:
        st.session_state.selected_product_id = products[0]["product_id"]
    if "recommend_session_id" not in st.session_state:
        st.session_state.recommend_session_id = None
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "latest_intent" not in st.session_state:
        st.session_state.latest_intent = None
    if "latest_bridged" not in st.session_state:
        st.session_state.latest_bridged = None
    if "latest_locations" not in st.session_state:
        st.session_state.latest_locations = []
    if "offline_mode" not in st.session_state:
        st.session_state.offline_mode = False
    if "selected_persona_avatar" not in st.session_state:
        st.session_state.selected_persona_avatar = personas[0].get("avatar", "🙂")
    if "selected_persona_name" not in st.session_state:
        st.session_state.selected_persona_name = personas[0].get("persona", {}).get("name", "Persona")


def reset_recommendation_conversation() -> None:
    st.session_state.recommend_session_id = None
    st.session_state.chat_messages = []
    st.session_state.latest_intent = None
    st.session_state.latest_bridged = None
    st.session_state.latest_locations = []


def stream_chat_response(text: str):
    if not text:
        yield text
        return

    chunk_size = 24
    for index in range(0, len(text), chunk_size):
        yield text[: index + chunk_size]


def format_intent_label(intent: dict[str, Any] | str | None) -> str:
    if isinstance(intent, dict):
        raw_value = intent.get("label") or intent.get("intent") or intent.get("name")
    else:
        raw_value = intent

    if not raw_value:
        return "No intent yet"

    return str(raw_value).replace("_", " ").strip().title()


def format_bridge_label(bridge_category: str | None) -> str:
    if not bridge_category:
        return "location recommendation"
    return str(bridge_category).replace("_", " ").strip().title()


def format_location_label(location: Any) -> str:
    if isinstance(location, dict):
        return str(
            location.get("name")
            or location.get("venue_name")
            or location.get("venue_id")
            or location.get("title")
            or "Suggested venue"
        )
    return str(location)


def select_persona() -> dict[str, Any]:
    personas = load_personas()

    st.sidebar.title("Persona First")
    st.sidebar.caption("Pick a judge persona and every API call will use it automatically.")

    labels = [f"{entry['avatar']} {entry['name']}" for entry in personas]
    default_index = next((i for i, entry in enumerate(personas) if entry["id"] == st.session_state.selected_persona_id), 0)
    chosen_label = st.sidebar.selectbox("Select a persona", labels, index=default_index)
    chosen_entry = next(entry for entry in personas if f"{entry['avatar']} {entry['name']}" == chosen_label)

    if chosen_entry["id"] != st.session_state.selected_persona_id:
        st.session_state.selected_persona_id = chosen_entry["id"]
        st.session_state.selected_persona = chosen_entry["persona"]
        st.session_state.selected_persona_avatar = chosen_entry.get("avatar", "🙂")
        st.session_state.selected_persona_name = chosen_entry["persona"].get("name", "Persona")
    else:
        st.session_state.selected_persona_avatar = chosen_entry.get("avatar", "🙂")
        st.session_state.selected_persona_name = chosen_entry["persona"].get("name", "Persona")

    st.sidebar.markdown("---")
    st.sidebar.markdown(persona_card(chosen_entry))
    st.sidebar.checkbox(
        "Offline Mode",
        key="offline_mode",
        help="Use cached demo responses instead of live API calls.",
    )
    st.sidebar.markdown("---")
    st.sidebar.json(chosen_entry["persona"], expanded=False)

    return chosen_entry["persona"]


def post_json(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(f"{DEFAULT_BACKEND_URL}{path}", json=payload, timeout=600)
    response.raise_for_status()
    return response.json()


def build_payload(base_payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(base_payload)
    if st.session_state.offline_mode:
        payload["demo_mode"] = True
    return payload


def chat_avatar(role: str) -> str:
    if role == "assistant":
        return st.session_state.get("selected_persona_avatar", "🙂")
    return "🙂"


def render_chat_message(role: str, content: str) -> None:
    with st.chat_message(role, avatar=chat_avatar(role)):
        st.write(content)


def render_simulator_tab(persona: dict[str, Any]) -> None:
    st.subheader("Review Simulator")
    st.write("Generate a Nigerian-voice review for any product using the selected persona.")

    products = load_review_products()
    product_labels = [
        f"{product['category']} | {product['title']} | {product['product_id']}"
        for product in products
    ]
    default_index = next(
        (index for index, product in enumerate(products) if product["product_id"] == st.session_state.get("selected_product_id")),
        0,
    )

    with st.form("simulate_form", clear_on_submit=False):
        selected_label = st.selectbox("Choose a product", product_labels, index=default_index)
        selected_product = next(product for product in products if f"{product['category']} | {product['title']} | {product['product_id']}" == selected_label)
        submitted = st.form_submit_button("Generate review")

    if submitted:
        st.session_state.selected_product_id = selected_product["product_id"]
        payload = build_payload({"persona_description": persona, "product_id": selected_product["product_id"]})
        with st.spinner("Calling /simulate..."):
            result = post_json("/simulate", payload)
        st.success("Review generated")
        rating = int(result["rating"])
        stars = "★" * rating + "☆" * (5 - rating)
        st.markdown(f"### {stars}  `{rating}/5`")
        st.markdown(result["review"])
        with st.expander("Request payload", expanded=False):
            st.code(json.dumps(payload, indent=2, ensure_ascii=False), language="json")


def render_recommender_tab(persona: dict[str, Any]) -> None:
    st.subheader("Conversational Recommender")
    st.write("Keep asking follow-up questions. The same persona stays attached to each turn.")

    control_left, control_right = st.columns([1, 4])
    with control_left:
        if st.button("New Conversation", use_container_width=True):
            reset_recommendation_conversation()
            st.rerun()
    with control_right:
        if st.session_state.recommend_session_id:
            st.caption(f"Session ID: {st.session_state.recommend_session_id}")
        else:
            st.caption("Session ID will be created on the first message.")

    chat_col, reason_col = st.columns([3, 1.15], gap="large")

    with chat_col:
        for message in st.session_state.chat_messages:
            render_chat_message(message["role"], message["content"])

    with reason_col:
        with st.expander("Intent Reasoning", expanded=True):
            intent = st.session_state.latest_intent or {}
            intent_label = format_intent_label(intent)
            confidence = intent.get("confidence") if isinstance(intent, dict) else None
            bridge_category = intent.get("bridge_category") if isinstance(intent, dict) else None

            st.markdown(f"**Detected Intent:** {intent_label}")
            if confidence is not None:
                try:
                    confidence_text = f"{round(float(confidence) * 100)}%"
                except Exception:
                    confidence_text = str(confidence)
            else:
                confidence_text = "Unknown"
            st.markdown(f"**Confidence:** {confidence_text}")

            bridged = st.session_state.latest_bridged
            if bridged is None:
                st.markdown("**Bridge triggered:** Not yet")
            elif bridged:
                bridge_text = format_bridge_label(bridge_category)
                st.markdown(f"**Bridge triggered:** Yes → {bridge_text}")
            else:
                st.markdown("**Bridge triggered:** No")

            if st.session_state.latest_locations:
                st.markdown("**Latest locations:**")
                for location in st.session_state.latest_locations:
                    st.caption(f"- {format_location_label(location)}")
            else:
                st.caption("Location suggestions will appear here when a bridge is triggered.")

    user_message = st.chat_input("Ask for a product, then ask for a place to use it...")
    if user_message:
        st.session_state.chat_messages.append({"role": "user", "content": user_message})
        render_chat_message("user", user_message)

        payload = {
            "persona_description": persona,
            "message": user_message,
            "session_id": st.session_state.recommend_session_id,
        }
        payload = build_payload(payload)
        with st.spinner("Calling /recommend..."):
            result = post_json("/recommend", payload)

        st.session_state.recommend_session_id = result["session_id"]
        assistant_text = result["recommendation"]
        st.session_state.latest_intent = result.get("intent") or result.get("debug", {}).get("intent")
        st.session_state.latest_bridged = result.get("debug", {}).get("bridged")
        st.session_state.latest_locations = result.get("debug", {}).get("locations") or []
        st.session_state.chat_messages.append({"role": "assistant", "content": assistant_text})

        with st.chat_message("assistant", avatar=chat_avatar("assistant")):
            if hasattr(st, "write_stream"):
                st.write_stream(stream_chat_response(assistant_text))
            else:
                st.write(assistant_text)
            with st.expander("Debug state", expanded=False):
                st.json(result["debug"])


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🇳🇬", layout="wide")
    st.markdown("## OmniNaija Intent Graph | Persona-aware recommendations, reviews, and bridge suggestions")

    init_state()
    persona = select_persona()

    tab_simulate, tab_recommend = st.tabs(["Review Simulator", "Conversational Recommender"])
    with tab_simulate:
        render_simulator_tab(persona)
    with tab_recommend:
        render_recommender_tab(persona)


if __name__ == "__main__":
    main()
