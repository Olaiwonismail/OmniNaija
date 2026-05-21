from google import genai

from config import Config


def run_sanity_check():
    if not Config.GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not set in environment.")
        return

    client = genai.Client(api_key=Config.GEMINI_API_KEY)

    print(f"Calling model: {Config.GEMINI_MODEL}...")
    try:
        response = client.models.generate_content(
            model=Config.GEMINI_MODEL,
            contents="Say hello in one short sentence.",
        )
        print("Response:", response.text)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_sanity_check()
