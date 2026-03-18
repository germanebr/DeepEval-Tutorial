from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

from models.gcp_gemini import gcp_gemini_eval_model

def get_ans_relevancy_score(user_input:str, generated_ans:str):
    test_case = LLMTestCase(
        input=user_input,
        actual_output=generated_ans
    )

    metric = AnswerRelevancyMetric(
        threshold=0.5, # Passing threshold, default is 0.5
        model=gcp_gemini_eval_model, # DeepEval model that will perform as the LLM-as-a-judge
        include_reason=True,    # Includes a reason for the the evaluation score
        strict_mode=False,  # When True, the metric score will only have a total value of 0 or 1
        verbose_mode=False,  # When True, prints the intermediate steps used to calculate the score
    )
    metric.measure(test_case)
    return metric