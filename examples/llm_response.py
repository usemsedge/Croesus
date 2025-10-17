import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from audio import get_llm_response

if __name__ == "__main__":
    text = "I can't see my package delivery status. Can you help?"
    attributes = {"accent": "Indian", "gender": "female"}
    response = get_llm_response(text, attributes)
    print("LLM Response:", response)