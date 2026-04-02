from deepeval.prompt import Prompt
from deepeval.test_case import LLMTestCase

from prompt_optimization.algorithms.gepa import gepa_optimizer
from prompt_optimization.algorithms.mipro import mipro_optimizer

from models.gcp_gemini import GCP_GENERATION_MODEL

class MyPromptOptimizer:
    def __init__(self, prompt:str, goldens:list[LLMTestCase], metrics:list):
        self.prompt = Prompt(text_template=prompt)
        self.goldens = goldens
        self.llm = GCP_GENERATION_MODEL()
        self.metrics = metrics
    
    def _model_callback(self, prompt:Prompt, golden) -> str:
        prompt_to_llm = prompt.interpolate(input=golden.input)

        return self.llm.generate(prompt_to_llm)
    
    def _gepa_opt(self) -> str:
        """
        This algorithm is better when the prompt needs to maintain diversity across different problems at the same time,
        or when the task does not benefit from few-shot examples.
        """

        optimizer = gepa_optimizer(
            model_callback=self._model_callback,
            metrics=self.metrics
        )

        optimized_prompt = optimizer.optimize(
            prompt=self.prompt,
            goldens=self.goldens    # Requires minimum 2 goldens to work
        )
        print(optimized_prompt)
        return optimized_prompt
    
    def _mipro_opt(self) -> str:
        """
        This algorithm is better when few-shots are important for the prompt, or when you have a large
        candidate space to explore and optimize.
        """

        optimizer = mipro_optimizer(
            model_callback=self._model_callback,
            metrics=self.metrics
        )

        optimized_prompt = optimizer.optimize(
            prompt=self.prompt,
            goldens=self.goldens    # Requires minimum 2 goldens to work
        )
        print(optimized_prompt)
        return optimized_prompt
    
    def optimize_prompt(self, algorithm:str="gepa"):
        if algorithm == "mipro":
            print("Optimizing through MIPROv2")
            return self._mipro_opt()
        else:
            print("Optimizing through GEPA")
            return self._gepa_opt()