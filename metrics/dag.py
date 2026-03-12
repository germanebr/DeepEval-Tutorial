"""
The Deep Acyclic Graph (DAG) is one of the most flexible metrics for building deterministic decision trees to evaluate LLM outputs using LLMs-as-a-judge.

DAG gives more deterministic control over GEval, although you can use GEval or any other default metrics within the DAG metric.
Just like other LLM-as-a-judge metrics, DAG requires the context input and the GenAI-generated content to perform the evaluation.

The DAG structure must have direct edges and cannot be cyclic. It will be composed by four different node types:

* Task Node: It processes the test case into the judge's required format
* Binary Judgement Node: Based on a given criteria, this node will generate a True/False statement based on whether the criteria is met or not
* Non Binary Judgement Node: Based on a given criteria, this node will generate a non-deterministic verdict with a different value from the True/False statement
* Verdict Node: This will ALWAYS be the leaf node of the DAG, and determines the final output score based on the evaluation path

Additional information found in https://deepeval.com/docs/metrics-dag
"""

from deepeval.test_case import LLMTestCaseParams, LLMTestCase
from deepeval.metrics import DAGMetric, GEval
from deepeval.metrics.dag import (
    DeepAcyclicGraph,
    TaskNode,
    BinaryJudgementNode,
    NonBinaryJudgementNode,
    VerdictNode
)

from models.gcp_gemini import gcp_gemini_eval_model

def _build_task_node(instructions:str, evaluation_params:list[LLMTestCaseParams], output_label:str, children:list[VerdictNode|BinaryJudgementNode|NonBinaryJudgementNode]) -> TaskNode:
    return TaskNode(
        instructions=instructions,
        evaluation_params=evaluation_params,
        output_label=output_label,
        children=children
    )

def _build_verdict_node(verdict:str|bool, param:GEval|VerdictNode|BinaryJudgementNode|NonBinaryJudgementNode|int) -> VerdictNode:
    # If the parameter has a score, it's one of the last nodes from the DAG
    if isinstance(param, int):
        return VerdictNode(
            verdict=verdict,
            score=param
        )
    
    # If the parameter has a child node, it must specify which node will be used to continue with the DAG
    else:
        return VerdictNode(
            verdict=verdict,
            child=param
        )

def _build_nonbinary_judgement_node(criteria:str, children:list[dict[str, int|NonBinaryJudgementNode|BinaryJudgementNode]]) -> NonBinaryJudgementNode:
    # Build the Verdict Nodes needed
    verdict_nodes = [_build_verdict_node(i['verdict'], i['param']) for i in children]

    return NonBinaryJudgementNode(
        criteria=criteria,
        children=verdict_nodes
    )

def _build_binary_judgement_node(criteria:str, children:list[dict[bool, int|NonBinaryJudgementNode|BinaryJudgementNode]]) -> BinaryJudgementNode:
    # Build the Verdict Nodes needed
    verdict_nodes = [_build_verdict_node(i['verdict'], i['param']) for i in children]

    return BinaryJudgementNode(
        criteria=criteria,
        children=verdict_nodes
    )

def _build_dag():
    # We'll create a GEval to include in the evaluation example
    final_metric = GEval(
        name="Summary Correctness",
        criteria="Assign an integer value between 6 and 10 to measure the quality of the generated summary",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=gcp_gemini_eval_model,
        threshold=0.5,
        strict_mode=False,
        verbose_mode=False,
    )

    # Build your DAG structure to evaluate, from bottom to top
    correct_order_node = _build_nonbinary_judgement_node(
        criteria="Are the summary headings in the correct order: 'intro' => 'body' => 'conclusion'?",
        children=[
            {"verdict": 'Yes', "param": final_metric},
            {"verdict": 'Two are out of order', "param": 4},
            {"verdict": 'All are out of order', "param": 2}        
        ]
    )

    correct_headings_node = _build_binary_judgement_node(
        criteria="Does the summary headings contain all three: 'intro', 'body', and 'conclusion'?",
        children=[
            {"verdict": True, "param": correct_order_node},
            {"verdict": False, "param": 0}
        ]
    )

    extract_headings_node = _build_task_node(
        instructions="Extract all headings in `actual_output`",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        output_label="Summary headings",
        children=[correct_headings_node, correct_order_node]
    )

    dag = DeepAcyclicGraph(root_nodes=[extract_headings_node])
    return dag

def get_dag_score(user_input:str, generated_ans:str, metric_name:str):
    # Prepare the test case
    test_case = LLMTestCase(
        input=user_input,
        actual_output=generated_ans
    )

    # Get the DAG structure based on your use case
    dag = _build_dag()

    # Build the metric
    metric = DAGMetric(
        name=metric_name,
        dag=dag,
        threshold=0.5, # Passing threshold, default is 0.5
        model=gcp_gemini_eval_model, # DeepEval model that will perform as the LLM-as-a-judge
        include_reason=True, # Includes a justification for the decision
        strict_mode=False,  # When True, the metric score will only have a total value of 0 or 1
        verbose_mode=False,  # When True, prints the intermediate steps used to calculate the score
    )
    metric.measure(test_case)
    return metric