"""
G-Eval is a framework originally from the paper "NLG Evaluation using GPT-4 with Better Human Alignment" that uses LLMs to evaluate LLM outputs (aka. LLM-Evals),
and is one the best ways to create task-specific metrics.

The G-Eval algorithm first generates a series of evaluation steps for chain of thoughts (CoTs) prompting before using the generated steps to determine the
final score via a "form-filling paradigm" (which is just a fancy way of saying G-Eval requires different LLMTestCase parameters for evaluation depending
on the generated steps).

After generating a series of evaluation steps, G-Eval will:

1. Create prompt by concatenating the evaluation steps with all the parameters in an LLMTestCase that is supplied to evaluation_params.
2. At the end of the prompt, ask it to generate a score between 1–5, where 5 is better than 1.
3. Take the probabilities of the output tokens from the LLM to normalize the score and take their weighted summation as the final result.

Note that although GEval is great it many ways as a custom, task-specific metric, it is NOT deterministic. If you're looking for more fine-grained,
deterministic control over your metric scores, you should be using the DAGMetric instead.

The way G-Eval works is that the judge model first generates a series of evaluation steps, using CoTs, based on the given criteria. Once those steps
are created (or given), the judge model uses that information to determine the final score.

Additional information found in https://deepeval.com/docs/metrics-llm-evals
"""

import textwrap

from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric

from models.gcp_gemini import gcp_gemini_eval_model

from typing import Optional

def get_geval_score(user_input:str, generated_ans:str, expected_output:str,
                    metric_name:str, evaluation_parameters:list[LLMTestCaseParams], metric_criteria:Optional[str]=None, metric_steps:Optional[list[str]]=None, rubric:Optional[list[Rubric]]=None):
    test_case = LLMTestCase(
        input=user_input,
        actual_output=generated_ans,
        expected_output=expected_output
    )

    if metric_criteria:
        print("\t* Using criteria instead of evaluation steps")
        metric_steps = None
    elif metric_steps:
        print("\t* Using evaluation steps instead of criteria")
    else:
        raise Exception("You must provide either a criteria or a list with evaluation steps to perform this metric.")

    metric = GEval(
        name=metric_name,   # Name of the custom metric
        criteria=metric_criteria,   # A description outlining the specific evaluation aspects for each test case
        evaluation_params=evaluation_parameters,    # The parameters that are relevant for evaluation
        evaluation_steps=metric_steps,  # A list of strings outlining the exact steps the LLM should take for evaluation. Do NOT use if you give the criteria parameter
        rubric=rubric,  # A list of Rubrics that allows you to confine the range of the final metric score
        threshold=0.5, # Passing threshold, default is 0.5
        model=gcp_gemini_eval_model, # DeepEval model that will perform as the LLM-as-a-judge
        strict_mode=False,  # When True, the metric score will only have a total value of 0 or 1
        verbose_mode=False,  # When True, prints the intermediate steps used to calculate the score
    )
    metric.measure(test_case)
    return metric