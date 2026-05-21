from config import Config
from llm import generate_text_with_fallback


def run_sanity_check():
    if not Config.GROQ_API_KEY and not Config.GEMINI_API_KEY:
        print("Error: neither GROQ_API_KEY nor GEMINI_API_KEY is set in environment.")
        return

    try:
        response_text, provider = generate_text_with_fallback("Say hello in one short sentence.")
        print(f"Response provider: {provider}")
        print("Response:", response_text)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_sanity_check()
