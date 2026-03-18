import sys
import json

from google.genai.types import Part

from pathlib import Path
from dotenv import load_dotenv

# Resolve the repo root from this file's location
repo_root_path = Path(__file__).resolve().parents[2]
sys.path.append(str(repo_root_path))

env_path = repo_root_path / ".env"
load_dotenv(env_path)

from models.gcp_gemini import GCP_GENERATION_MODEL
    
def generate_rag_context(filepath:str, prompt:str, structure:dict):
    # Get the document's bytes to send to the LLM
    with open(filepath, "rb") as f:
        doc_bytes = f.read()
    
    # Extract the content with the LLM in the given format
    pdf_file = Part.from_bytes(
        data=doc_bytes,
        mime_type="application/pdf"
    )

    llm = GCP_GENERATION_MODEL()
    response = llm.generate(
        prompt=prompt,
        context=pdf_file,
        response_schema=structure
    )

    # Generate the json file of the document
    parsed_response = json.loads(response)
    with open("./docs/rag_context_sample.json", "w", encoding="utf-8") as f:
        json.dump(parsed_response, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    prompt = "Extract the document's content and divide it into their corresponding sections."

    structure = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "section_title": {"type": "STRING"},
                "section_content": {"type": "STRING"}
            },
            "required": ["section_title", "section_content"]
        }
    }

    generate_rag_context("./docs/WHO_statistics_2025.pdf", prompt, structure)