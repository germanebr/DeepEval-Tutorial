from google import genai
from google.genai.types import HttpOptions
from google.oauth2 import service_account

from custom import config

from deepeval.models.base_model import DeepEvalBaseLLM

GEMINI_MODEL = os.getenv("GEMINI_MODEL_NAME")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
GOOGLE_GENAI_USER_VERTEXAI = os.getenv("GOOGLE_GENAI_USER_VERTEXAI")

class GoogleVertexAI(DeepEvalBaseLLM):
    """Class to implement Vertex AI for DeepEval"""
    def __init__(self):
        self.model = genai.Client(
            credentials=self._get_credentials(),
            http_options=HttpOptions(api_version="v1"),
            vertexai=GOOGLE_GENAI_USER_VERTEXAI,
            project=GOOGLE_CLOUD_PROJECT,
            location=GOOGLE_CLOUD_LOCATION
        )

    def _get_credentials(self):
        return service_account.Credentials.from_service_account_file(
            filename=config.GOOGLE_APPLICATION_CREDENTIALS,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

    def load_model(self):
        return self.model

    def generate(self, prompt:str, context:str='', response_schema:dict=None) -> str:
        if response_schema:
            struct_out = {
                "response_mime_type": "application/json",
                "response_schema": response_schema
            }
        else:
            struct_out = None
        
        response = self.model.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt, context],
            config=struct_out
        )
        return response.text

    async def a_generate(self, prompt:str, context:str='', response_schema:dict=None) -> str:
        return self.generate(prompt, context, response_schema)

    def get_model_name(self):
        return "GCP Vertex AI Model"