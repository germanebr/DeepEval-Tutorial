from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig

from models.deepeval_gemini_llm import GoogleVertexAI
from models.deepeval_gemini_embedder import CustomEmbeddingModel

from custom import config

class GoldenGenerator:
    def __init__(self):
        self.llm = GoogleVertexAI()
        self.embedder = CustomEmbeddingModel()
        self.synthesizer = Synthesizer(model=self.llm)

    def generate_goldens(self, doc_paths:list[str]):
        goldens = self.synthesizer.generate_goldens_from_docs(
            document_paths = doc_paths,
            include_expected_output = True,
            max_goldens_per_context = config.max_goldens_per_context,
            context_construction_config = ContextConstructionConfig(
                critic_model = self.llm,
                max_contexts_per_document = config.max_contexts_per_document,
                min_contexts_per_document = config.min_contexts_per_document,
                max_context_length = config.max_context_length,
                min_context_length = config.min_context_length,
                chunk_size = config.chunk_size,
                chunk_overlap = config.chunk_overlap,
                context_quality_threshold = config.context_quality_threshold,
                context_similarity_threshold = config.context_similarity_threshold,
                max_retries = config.max_retries,
                embedder = self.embedder
            )
        )
        return goldens