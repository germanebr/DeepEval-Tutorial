import os
import json

from deepeval.models import GeminiModel

from google import genai
from google.genai.types import HttpOptions

from google.oauth2 import service_account

from dotenv import load_dotenv

load_dotenv("./.env")

GEMINI_MODEL = os.getenv("GEMINI_MODEL_NAME")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
GOOGLE_GENAI_USER_VERTEXAI = os.getenv("GOOGLE_GENAI_USER_VERTEXAI")

# Read the json and set it on the service account key
with open(GOOGLE_APPLICATION_CREDENTIALS) as f:
    gcp_service_account_key = json.load(f)

gcp_gemini_eval_model = GeminiModel(
    model=GEMINI_MODEL,
    project=GOOGLE_CLOUD_PROJECT,
    location=GOOGLE_CLOUD_LOCATION,
    service_account_key=gcp_service_account_key,
    temperature=0
)

class GCP_GENERATION_MODEL:
    def __init__(self):
        self.client = genai.Client(
            credentials=self._get_credentials(),
            http_options=HttpOptions(api_version="v1"),
            vertexai=GOOGLE_GENAI_USER_VERTEXAI,
            project=GOOGLE_CLOUD_PROJECT,
            location=GOOGLE_CLOUD_LOCATION
        )

    def _get_credentials(self):
        return service_account.Credentials.from_service_account_file(
            filename=GOOGLE_APPLICATION_CREDENTIALS,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

    def generate(self, prompt:str, context:str):
        response = self.client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt, context]
        )
        return response.text