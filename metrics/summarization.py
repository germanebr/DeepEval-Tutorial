"""
The summarization metric uses an LLM-as-a-judge to determine whether the AI-generated content factually aligns and
includes the necessary details from the original given text.

In DeepEval, the input belongs to the original text, while the actual_output refers to the AI-generated content.

This metric is obtained by getting the minimum value between the Alignment Score and the Coverage Score.

The Alignment Score determines whether the summary contains hallucinated or contradictory information to the original text.

The Coverage Score determines whether the summary contains the necessary information from the original text.
It evaluates n closed-ended (Yes or No) questions to obtain a ratio of which the original text and summary yields the same answer.

Additional information found in https://deepeval.com/docs/metrics-summarization
"""

from deepeval.test_case import LLMTestCase
from deepeval.metrics import SummarizationMetric

from models.gcp_gemini import gcp_gemini_eval_model

def get_summary_score(original_text:str, summary:str):
    test_case = LLMTestCase(
        input=original_text,
        actual_output=summary
    )

    metric = SummarizationMetric(
        threshold=0.5, # Passing threshold, default is 0.5
        model=gcp_gemini_eval_model, # DeepEval model that will perform as the LLM-as-a-judge
        assessment_questions=[  # Close-ended questions that can be used with yes or no
            "Is the coverage score based on a percentage of 'yes' answers?",
            "Does the score ensure the summary's accuracy with the source?",
            "Does a higher score mean a more comprehensive summary?"
        ],
        include_reason=True,    # Includes a reason for the the evaluation score
        strict_mode=False,  # When True, the metric score will only have a total value of 0 or 1
        verbose_mode=False,  # When True, prints the intermediate steps used to calculate the score
        truths_extraction_limit=None    # The maximum number of factual truths to extract from the original text to determine the alignment score
    )
    metric.measure(test_case)
    return metric