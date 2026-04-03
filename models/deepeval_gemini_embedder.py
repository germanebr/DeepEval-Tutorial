from google import genai
from google.genai.types import HttpOptions
from google.oauth2 import service_account

from deepeval.models import DeepEvalBaseEmbeddingModel

from custom import config

class CustomEmbeddingModel(DeepEvalBaseEmbeddingModel):
    def __init__(self):
        self.model = genai.Client(
            credentials=self._get_credentials(),
            http_options=HttpOptions(api_version="v1"),
            vertexai=config.GOOGLE_GENAI_USER_VERTEXAI,
            project=config.GOOGLE_CLOUD_PROJECT,
            location=config.GOOGLE_CLOUD_LOCATION
        )

    def _get_credentials(self):
        return service_account.Credentials.from_service_account_file(
            filename=config.GOOGLE_APPLICATION_CREDENTIALS,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

    def load_model(self):
        return self.model

    def embed_text(self, text: str) -> list[float]:
        res = self.model.embed_content(
            model=config.GEMINI_EMBEDDING,
            contents=text
        )
        return res.embeddings[0].values

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        res = self.model.models.embed_content(
            model=config.GEMINI_EMBEDDING,
            contents=texts
        )
        return [x.values for x in res.embeddings]

    async def a_embed_text(self, text: str) -> list[float]:
        return self.embed_text(text)
    
    async def a_embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.embed_texts(texts)

    def get_model_name(self):
        "Custom GCP Vertex AI Gemini Embedding Model"