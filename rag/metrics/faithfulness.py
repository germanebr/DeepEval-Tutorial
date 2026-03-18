"""
Faithfulness measures the quality of the RAG by confirming that the generated content
factually aligns with the retrieval context.

This metric might sound similar to the Custom Hallucination metric, but this one is more
focused on detecting the contradictions betwee the generated content and the retrieval context.

Additional information found in https://deepeval.com/docs/metrics-faithfulness
"""

from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from models.gcp_gemini import gcp_gemini_eval_model

def get_faithfulness_score(user_input:str, generated_ans:str, retrieved_context:list[str]) -> FaithfulnessMetric:
    test_case = LLMTestCase(
        input=user_input,
        actual_output=generated_ans,
        retrieval_context=retrieved_context
    )

    metric = FaithfulnessMetric(
        threshold=0.5, # Passing threshold, default is 0.5
        model=gcp_gemini_eval_model, # DeepEval model that will perform as the LLM-as-a-judge
        include_reason=True,    # Includes a reason for the the evaluation score
        strict_mode=False,  # When True, the metric score will only have a total value of 0 or 1
        verbose_mode=False,  # When True, prints the intermediate steps used to calculate the score
        truths_extraction_limit=None,    # The maximum number of factual truths to extract from the retrieved_contex, ordered by importance
        penalize_ambiguous_claims=False # Parameter when in True, will not consider claims that are ambiguous as faithful
    )
    metric.measure(test_case)
    return metric