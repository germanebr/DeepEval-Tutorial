"""
Contextual relevancy measures the quality of the RAG's retrieved context by evaluating
the overall relevance of the whole context against the user input.

It divides the number of relevant statements found in the retrieved context against
the total number of statements retrieved.

Additional information found in https://deepeval.com/docs/metrics-contextual-relevancy
"""

from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRelevancyMetric

from models.gcp_gemini import gcp_gemini_eval_model

def get_contextual_relevancy_score(user_input:str, generated_ans:str, retrieval_context:list[str]) -> ContextualRelevancyMetric:
    test_case = LLMTestCase(
        input=user_input,
        actual_output=generated_ans,
        retrieval_context=retrieval_context
    )

    metric = ContextualRelevancyMetric(
        threshold=0.5, # Passing threshold, default is 0.5
        model=gcp_gemini_eval_model, # DeepEval model that will perform as the LLM-as-a-judge
        include_reason=True,    # Includes a reason for the the evaluation score
        strict_mode=False,  # When True, the metric score will only have a total value of 0 or 1
        verbose_mode=False,  # When True, prints the intermediate steps used to calculate the score
    )
    metric.measure(test_case)
    return metric