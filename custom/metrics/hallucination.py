"""
The hallucination metric uses LLM-as-a-judge to dtermine whether the LLM generates factually correct information by comparing the generated content to the provided context.

It is important to mention that this metric is used for hallucinations on generated content, not for RAG information retrieval. For that, there's a different metric (faithfulness).

The hallucination metric is obtained by dividing the number of contradicted contexts in the generated answer, by the total number of contexts provided to the model.

Additional information found in https://deepeval.com/docs/metrics-hallucination
"""

from deepeval.test_case import LLMTestCase
from deepeval.metrics import HallucinationMetric

from models.gcp_gemini import gcp_gemini_eval_model

def get_hallucination_score(context:list[str], user_input:str, generated_ans:str):
    test_case = LLMTestCase(
        input=user_input,
        actual_output=generated_ans,
        context=context
    )

    metric = HallucinationMetric(
        threshold=0.5, # Passing threshold, default is 0.5
        model=gcp_gemini_eval_model, # DeepEval model that will perform as the LLM-as-a-judge
        include_reason=True,    # Includes a reason for the the evaluation score
        strict_mode=False,  # When True, the metric score will only have a total value of 0 or 1
        verbose_mode=False,  # When True, prints the intermediate steps used to calculate the score
    )
    metric.measure(test_case)
    return metric