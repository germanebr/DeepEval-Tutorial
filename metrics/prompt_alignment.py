"""
The prompt alignment is used to determine whether the LLM is able to generate content that aligns with the instructions specified in the model's prompt.

It uses an LLM-as-a-judge to perform the evaluation.
This metric is obtained by dividing the number of instructions followed or considered in the generated answer, with the total number of instructions present in the given prompt.

The LLM judge classifies whether each instruction is followed in the generated answer, so it will generate better metrics if we pass directly the instructions rather than the whole prompt.

Additional information found in https://deepeval.com/docs/metrics-prompt-alignment
"""

from deepeval.test_case import LLMTestCase
from deepeval.metrics import PromptAlignmentMetric

from models.gcp_gemini import gcp_gemini_eval_model

def get_prompt_alignment_score(prompt:str, user_input:str, generated_ans:str):
    test_case = LLMTestCase(
        input=user_input,
        actual_output=generated_ans
    )

    metric = PromptAlignmentMetric(
        prompt_instructions=[prompt], # A list of strings specifying the instructions you want followed in the prompt template.
        threshold=0.5, # Passing threshold, default is 0.5
        model=gcp_gemini_eval_model, # DeepEval model that will perform as the LLM-as-a-judge
        include_reason=True,    # Includes a reason for the the evaluation score
        strict_mode=False,  # When True, the metric score will only have a total value of 0 or 1
        verbose_mode=False,  # When True, prints the intermediate steps used to calculate the score
    )
    metric.measure(test_case)
    return metric