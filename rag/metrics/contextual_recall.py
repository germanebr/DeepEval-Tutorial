"""
Contextual recall measures the quality of the RAG's retrieved context by evaluating the extent
of which the context aligns with the expected output.

Additional information found in https://deepeval.com/docs/metrics-contextual-recall
"""

from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric

from models.gcp_gemini import gcp_gemini_eval_model

def get_contextual_recall_score(user_input:str, generated_ans:str, expected_output:str, retrieval_context:list[str]) -> ContextualRecallMetric:
    test_case = LLMTestCase(
        input=user_input,
        actual_output=generated_ans,
        expected_output=expected_output,
        retrieval_context=retrieval_context
    )

    metric = ContextualRecallMetric(
        threshold=0.5, # Passing threshold, default is 0.5
        model=gcp_gemini_eval_model, # DeepEval model that will perform as the LLM-as-a-judge
        include_reason=True,    # Includes a reason for the the evaluation score
        strict_mode=False,  # When True, the metric score will only have a total value of 0 or 1
        verbose_mode=False,  # When True, prints the intermediate steps used to calculate the score
    )
    metric.measure(test_case)
    return metric