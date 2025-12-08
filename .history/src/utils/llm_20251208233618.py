from openai import OpenAI
import os

def get_llm_client():
    """
    Returns an initialized OpenAI client.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not found in environment variables.")
        return None
    return OpenAI(api_key=api_key)
