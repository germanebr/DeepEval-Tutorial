from deepeval.prompt import Prompt
from deepeval.metrics import AnswerRelevancyMetric, PromptAlignmentMetric
from prompt_optimization.algorithms.gepa import gepa_optimizer

from models.gcp_gemini import GCP_GENERATION_MODEL

class PromptOptimizer:
    def __init__(self, prompt:str, goldens:list[str]):
        self.prompt = Prompt(text_template=prompt)
        self.goldens = goldens
        self.llm = GCP_GENERATION_MODEL()
    
    def _model_callback(self, golden) -> str:
        prompt_to_llm = self.prompt.interpolate(input=golden.input)

        return self.llm.generate(prompt_to_llm)
    
    def optimize