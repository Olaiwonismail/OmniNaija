import json
import os
from pathlib import Path
from typing import Any

import requests
import streamlit as st


APP_TITLE = "OmniNaija Intent Graph"
DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
PERSONAS_PATH = Path(__file__).resolve().parent.parent / "personas" / "personas.json"


@st.cache_data
def load_personas() -> list[dict[str, Any]]:
    return json.loads(PERSONAS_PATH.read_text(encoding="utf-8"))


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
    if "selected_persona_id" not in st.session_state:
        st.session_state.selected_persona_id = personas[0]["id"]
    if "selected_persona" not in st.session_state:
        st.session_state.selected_persona = personas[0]["persona"]
    if "recommend_session_id" not in st.session_state:
        st.session_state.recommend_session_id = None
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []


def select_persona() -> dict[str, Any]:
    personas = load_personas()
    options = {entry["id"]: entry for entry in personas}

    st.sidebar.title("Persona First")
    st.sidebar.caption("Pick a judge persona and every API call will use it automatically.")

    labels = [f"{entry['avatar']} {entry['name']}" for entry in personas]
    default_index = next((i for i, entry in enumerate(personas) if entry["id"] == st.session_state.selected_persona_id), 0)
    chosen_label = st.sidebar.selectbox("Select a persona", labels, index=default_index)
    chosen_entry = next(entry for entry in personas if f"{entry['avatar']} {entry['name']}" == chosen_label)

    if chosen_entry["id"] != st.session_state.selected_persona_id:
        st.session_state.selected_persona_id = chosen_entry["id"]
        st.session_state.selected_persona = chosen_entry["persona"]

    st.sidebar.markdown("---")
    st.sidebar.markdown(persona_card(chosen_entry))
    st.sidebar.markdown("---")
    st.sidebar.json(chosen_entry["persona"], expanded=False)

    return chosen_entry["persona"]


def post_json(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(f"{DEFAULT_BACKEND_URL}{path}", json=payload, timeout=600)
    response.raise_for_status()
    return response.json()


def render_simulator_tab(persona: dict[str, Any]) -> None:
    st.subheader("Review Simulator")
    st.write("Generate a Nigerian-voice review for any product using the selected persona.")

    with st.form("simulate_form", clear_on_submit=False):
        product_id = st.text_input("Product ID / ASIN", value="B000BRLXUI")
        submitted = st.form_submit_button("Generate review")

    if submitted:
        payload = {"persona_description": persona, "product_id": product_id}
        with st.spinner("Calling /simulate..."):
            result = post_json("/simulate", payload)
        st.success("Review generated")
        st.metric("Rating", result["rating"])
        st.write(result["review"])
        st.code(json.dumps(payload, indent=2, ensure_ascii=False), language="json")


def render_recommender_tab(persona: dict[str, Any]) -> None:
    st.subheader("Conversational Recommender")
    st.write("Keep asking follow-up questions. The same persona stays attached to each turn.")

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_message = st.chat_input("Ask for a product, then ask for a place to use it...")
    if user_message:
        st.session_state.chat_messages.append({"role": "user", "content": user_message})
        with st.chat_message("user"):
            st.write(user_message)

        payload = {
            "persona_description": persona,
            "message": user_message,
            "session_id": st.session_state.recommend_session_id,
        }
        with st.spinner("Calling /recommend..."):
            result = post_json("/recommend", payload)

        st.session_state.recommend_session_id = result["session_id"]
        assistant_text = result["recommendation"]
        st.session_state.chat_messages.append({"role": "assistant", "content": assistant_text})

        with st.chat_message("assistant"):
            st.write(assistant_text)
            with st.expander("Debug state", expanded=False):
                st.json(result["debug"])


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🇳🇬", layout="wide")
    st.title(APP_TITLE)
    st.caption("Pick a persona first, then use the two tabs to test reviews and conversational recommendations.")

    init_state()
    persona = select_persona()

    tab_simulate, tab_recommend = st.tabs(["Review Simulator", "Conversational Recommender"])
    with tab_simulate:
        render_simulator_tab(persona)
    with tab_recommend:
        render_recommender_tab(persona)


if __name__ == "__main__":
    main()
