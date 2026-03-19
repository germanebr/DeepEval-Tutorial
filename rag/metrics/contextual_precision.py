"""
Contextual precision uses an LLM-as-a-judge to evaluate whether the RAG's nodes in the
retrieval_context are relevant to the user's input and are ranked higher than the
irrelevant ones. In fewer words, it assesses the ranking order of the retrieved chunks.

Additional information found in https://deepeval.com/docs/metrics-contextual-precision
"""

from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualPrecisionMetric

from models.gcp_gemini import gcp_gemini_eval_model

def get_contextual_precision_score(user_input:str, generated_ans:str, expected_output:str, retrieval_context:list[str]) -> ContextualPrecisionMetric:
    test_case = LLMTestCase(
        input=user_input,
        actual_output=generated_ans,
        expected_output=expected_output,
        retrieval_context=retrieval_context
    )

    metric = ContextualPrecisionMetric(
        threshold=0.5, # Passing threshold, default is 0.5
        model=gcp_gemini_eval_model, # DeepEval model that will perform as the LLM-as-a-judge
        include_reason=True,    # Includes a reason for the the evaluation score
        strict_mode=False,  # When True, the metric score will only have a total value of 0 or 1
        verbose_mode=False,  # When True, prints the intermediate steps used to calculate the score
    )
    metric.measure(test_case)
    return metric