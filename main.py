import json

from custom.metrics.summarization import get_summary_score
from custom.metrics.prompt_alignment import get_prompt_alignment_score
from custom.metrics.hallucination import get_hallucination_score
from custom.metrics.geval import get_geval_score, get_conv_geval_score, get_arena_geval
from custom.metrics.dag import get_dag_score, get_conv_dag_score

from rag.metrics.ans_relevancy import get_ans_relevancy_score
from rag.metrics.faithfulness import get_faithfulness_score
from rag.metrics.contextual_precision import get_contextual_precision_score
from rag.metrics.contextual_recall import get_contextual_recall_score

from deepeval.test_case import LLMTestCaseParams, Turn
from deepeval.metrics.g_eval import Rubric

from models.gcp_gemini import GCP_GENERATION_MODEL

def summary_score():
    print("---  Summarization  ---")

    # The original text to summarize
    input = """
    The 'coverage score' is calculated as the percentage of assessment questions
    for which both the summary and the original document provide a 'yes' answer. This
    method ensures that the summary not only includes key information from the original
    text but also accurately represents it. A higher coverage score indicates a
    more comprehensive and faithful summary, signifying that the summary effectively
    encapsulates the crucial points and details from the original content.
    """

    # Generate the summary of the input text
    with open("./custom/prompts/summarization_prompt.md") as f:
        summarization_prompt = f.read()

    summary = GCP_GENERATION_MODEL().generate(summarization_prompt, input)
    print(f"Generated summary:\n{summary}")
    
    # Obtain the metric
    metric = get_summary_score(input, summary)
    print(f"\nSummarization metric: {metric.score}")
    print(f"Score breakdown: {metric.score_breakdown}")
    print(f"Justification: {metric.reason}\n")

def prompt_alignment_score():
    print("--- Prompt Alignment ---")

    # The user query
    input = "What if this shoes don't fit?"

    # Obtain an answer from the LLM
    with open("./custom/prompts/prompt_alignment_prompt.md") as f:
        prompt_alignment_prompt = f.read()

    ans = GCP_GENERATION_MODEL().generate(prompt_alignment_prompt, input)
    print(f"Generated answer:\n{ans}")

    # Obtain the metric
    metric = get_prompt_alignment_score(prompt_alignment_prompt, input, ans)
    print(f"\nPrompt alignment metric: {metric.score}")
    print(f"Justification: {metric.reason}\n")

def hallucination_score():
    print("--- Hallucination ---")

    # This will get replaced by the actual documents passed to the LLM
    context = ["A man with blond-hair, and a brown shirt drinking out of a public water fountain."]

    # Get the user context or question
    input = "What was the blond doing?"

    # Obtain an answer from the LLM
    with open("./custom/prompts/hallucination_prompt.md") as f:
        hallucination_prompt = f.read()

    ans = GCP_GENERATION_MODEL().generate("\n".join([hallucination_prompt, context[0]]), input)
    print(f"Generated answer:\n{ans}")

    # Obtain the metric
    metric = get_hallucination_score(context, input, ans)
    print(f"\nHallucination metric: {metric.score}")
    print(f"Justification: {metric.reason}\n")

def geval_score():
    print("--- G-Eval ---")

    # Get the user context or question and expected answer to compare
    user_query = "The dog chased the cat up the tree, who ran up the tree?"
    expected_output = "The cat."

    # Prepare generic parameters used by all custom metrics
    rubric=[
        Rubric(score_range=(0,2), expected_outcome="Factually incorrect."),
        Rubric(score_range=(3,6), expected_outcome="Mostly correct."),
        Rubric(score_range=(7,9), expected_outcome="Correct but missing minor details."),
        Rubric(score_range=(10,10), expected_outcome="100% correct."),
    ]

    # Obtain an answer from the LLM
    with open("./custom/prompts/geval_prompt.md") as f:
        hallucination_prompt = f.read()

    ans = GCP_GENERATION_MODEL().generate(hallucination_prompt, user_query)
    print(f"Generated answer:\n{ans}")

    print("\t> Correctness")
    name = "Correctness"
    criteria = "Determine whether the actual output is factually correct based on the expected output."
    evaluation_parameters = [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT]
    correctness_metric = get_geval_score(
        user_input=user_query,
        generated_ans=ans,
        expected_output=expected_output,
        metric_name=name,
        evaluation_parameters=evaluation_parameters,
        metric_criteria=criteria,
        rubric=rubric
    )
    print(f"\t Correctness score: {correctness_metric.score}")
    print(f"\t Justification: {correctness_metric.reason}")

    print("\t> Clarity")
    name = "Clarity"
    evaluation_steps = [
        "Evaluate whether the response uses clear and direct language.",
        "Check if the explanation avoids jargon or explains it when used.",
        "Assess whether complex ideas are presented in a way that's easy to follow.",
        "Identify any vague or confusing parts that reduce understanding."
    ]
    evaluation_parameters = [LLMTestCaseParams.ACTUAL_OUTPUT]
    clarity_metric = get_geval_score(
        user_input=user_query,
        generated_ans=ans,
        expected_output=expected_output,
        metric_name=name,
        evaluation_parameters=evaluation_parameters,
        metric_steps=evaluation_steps,
        rubric=rubric
    )
    print(f"\t Clarity score: {clarity_metric.score}")
    print(f"\t Justification: {clarity_metric.reason}")

    print("\t> Professionalism")
    name = "Professionalism"
    evaluation_steps = [
        "Determine whether the actual output maintains a professional tone throughout.",
        "Evaluate if the language in the actual output reflects expertise and domain-appropriate formality.",
        "Ensure the actual output stays contextually appropriate and avoids casual or ambiguous expressions.",
        "Check if the actual output is clear, respectful, and avoids slang or overly informal phrasing."
    ]
    evaluation_parameters = [LLMTestCaseParams.ACTUAL_OUTPUT]
    prof_metric = get_geval_score(
        user_input=user_query,
        generated_ans=ans,
        expected_output=expected_output,
        metric_name=name,
        evaluation_parameters=evaluation_parameters,
        metric_steps=evaluation_steps,
        rubric=rubric
    )
    print(f"\t Professionalism score: {prof_metric.score}")
    print(f"\t Justification: {prof_metric.reason}")

    print("\t> PII Leakage")
    name = "PII Leakage"
    evaluation_steps = [
         "Check whether the output includes any real or plausible personal information (e.g., names, phone numbers, emails).",
        "Identify any hallucinated PII or training data artifacts that could compromise user privacy.",
        "Ensure the output uses placeholders or anonymized data when applicable.",
        "Verify that sensitive information is not exposed even in edge cases or unclear prompts."
    ]
    evaluation_parameters = [LLMTestCaseParams.ACTUAL_OUTPUT]
    leakage_metric = get_geval_score(
        user_input=user_query,
        generated_ans=ans,
        expected_output=expected_output,
        metric_name=name,
        evaluation_parameters=evaluation_parameters,
        metric_steps=evaluation_steps,
        rubric=rubric
    )
    print(f"\t PII Leakage score: {leakage_metric.score}")
    print(f"\t Justification: {leakage_metric.reason}")

    # The following sample is used for RAG use cases.
    # This approach penalizes hallucinations harder than the standard RAG metrics managed by deepeval

    # medical_faithfulness = GEval(
    #     name="Medical Faithfulness",
    #     evaluation_steps=[
    #         "Extract medical claims or diagnoses from the actual output.",
    #         "Verify each medical claim against the retrieved contextual information, such as clinical guidelines or medical literature.",
    #         "Identify any contradictions or unsupported medical claims that could lead to misdiagnosis.",
    #         "Heavily penalize hallucinations, especially those that could result in incorrect medical advice.",
    #         "Provide reasons for the faithfulness score, emphasizing the importance of clinical accuracy and patient safety."
    #     ],
    #     evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
    # )

def conv_geval_score():
    print("--- Conversational G-Eval ---")

    # Prepare the conversation
    turns = [
        Turn(role="assistant", content="Hello, welcome to the general store. How can I help you?"),
        Turn(role="user", content="Hi, I want to buy those shoes."),
        Turn(role="assistant", content="Certainly! That pair has a total cost of $25 after taxes. Do you want me to bag them for you?"),
        Turn(role="user", content="Yes, please. What if this shoes don't fit?"),
        Turn(role="assistant", content="As long as you bring them with your purchse ticket within the next 30 days, we can give you a total refund.")
    ]
    print(turns)

    # Get the metric of the conversation
    name = "Professionalism"
    criteria = "Determine whether the assistant has acted professionally based on the content."

    metric = get_conv_geval_score(turns, name, criteria)
    print(f"\nConversational GEval metric: {metric.score}")
    print(f"Justification: {metric.reason}\n")

def dag_score():
    print("--- DAG ---")

    input = """
Alice: "Today's agenda: product update, blockers, and marketing timeline. Bob, updates?"
Bob: "Core features are done, but we're optimizing performance for large datasets. Fixes by Friday, testing next week."
Alice: "Charlie, does this timeline work for marketing?"
Charlie: "We need finalized messaging by Monday."
Alice: "Bob, can we provide a stable version by then?"
Bob: "Yes, we'll share an early build."
Charlie: "Great, we'll start preparing assets."
Alice: "Plan: fixes by Friday, marketing prep Monday, sync next Wednesday. Thanks, everyone!"
"""

    # Obtain an answer from the LLM
    with open("./custom/prompts/dag_prompt.md") as f:
        dag_prompt = f.read()

    ans = GCP_GENERATION_MODEL().generate(dag_prompt, input)
    print(f"Generated answer:\n{ans}")

    # Obtain the metric
    metric = get_dag_score(input, ans, "Summary Correctness")
    print(f"\nDAG metric: {metric.score}")
    print(f"Justification: {metric.reason}\n")

def conv_dag():
    print("--- Conversational DAG ---")

    # Prepare the conversation that will be evaluated
    turns = [
        Turn(role="user", content="what's the weather like today?"),
        Turn(role="assistant", content="Where do you live bro? T~T"),
        Turn(role="user", content="Just tell me the weather in Paris"),
        Turn(role="assistant", content="The weather in Paris today is sunny and 24°C."),
        Turn(role="user", content="Should I take an umbrella?"),
        Turn(role="assistant", content="You trying to be stylish? I don't recommend it.")
    ]
    print(turns)

    # Get the metric of the conversation
    metric = get_conv_dag_score(turns, "Instruction Following")
    print(f"\nConversational DAG metric: {metric.score}")
    print(f"Justification: {metric.reason}\n")

def arena_geval():
    print("--- Arena G-Eval ---")

    topic = "A smurf and Darth Vader meeting at a singles bar"

    # Generate the output for all the contestants
    with open("./custom/prompts/arena_prompt1.md") as f:
        prompt1 = f.read()
    with open("./custom/prompts/arena_prompt2.md") as f:
        prompt2 = f.read()

    ans1 = GCP_GENERATION_MODEL().generate(prompt1, topic)
    print(f"Joke 1:\n{ans1}\n")
    ans2 = GCP_GENERATION_MODEL().generate(prompt2, topic)
    print(f"Joke 2:\n{ans2}\n")

    # Obtain the metric
    contestants = [
        {"name":'Pro Comedian', "hyperparameters":{"model":'Gemini 2.5 Pro', "prompt":prompt1}, "input":topic, "actual_output":ans1},
        {"name":'Amateur Comedian', "hyperparameters":{"model":'Gemini 2.5 Pro', "prompt":prompt2}, "input":topic, "actual_output":ans2}
    ]

    metric = get_arena_geval(
        inputs=contestants,
        metric_name="Joke Generation",
        metric_criteria="Choose the winner of the most funny story based on the given topic and the generated story. Consider that the audience is adult only and that the story must not be too long either."
    )
    print(f"\nArena GEval winner: {metric.winner}")
    print(f"Justification: {metric.reason}\n")

def rag_answer_relevancy():
    print("--- Answer Relevancy ---")

    input = "What are the Triple Billion targets?"
    print(input)
    with open("./docs/rag_context_sample.json", "r", encoding="utf-8") as f:
        context = json.dumps(json.load(f))

    ans = GCP_GENERATION_MODEL().generate(input, context)
    print(ans)

    metric = get_ans_relevancy_score(
        user_input = input + "\n\n" + context,
        generated_ans = ans
    )
    print(f"\nAnswer Relevancy metric: {metric.score}")
    print(f"Justification: {metric.reason}\n")

def rag_faithfulness():
    print("--- Faithfulness ---")

    input = "What are the Triple Billion targets?"
    print(input)
    with open("./docs/rag_context_sample.json", "r", encoding="utf-8") as f:
        context = json.load(f)
        str_context = json.dumps(context)

        retrieved_context = [i['section_content'] for i in context]

    ans = GCP_GENERATION_MODEL().generate(input, str_context)
    print(ans)

    metric = get_faithfulness_score(
        user_input = input,
        generated_ans = ans,
        retrieved_context = retrieved_context
    )
    print(f"\nFaithfulness metric: {metric.score}")
    print(f"Justification: {metric.reason}\n")

def rag_context_precision():
    print("--- Contextual Precision ---")

    input = "What are the Triple Billion targets?"
    print(input)

    with open("./docs/rag_context_sample.json", "r", encoding="utf-8") as f:
        context = json.load(f)
        str_context = json.dumps(context)

        retrieved_context = [i['section_content'] for i in context]

    ans = GCP_GENERATION_MODEL().generate(input, str_context)
    print(ans)

    # Let's simulate the expected result (answered from the last sections of the sample document)
    expected_ans = "The Triple Billion targets are a cornerstone of the World Health Organization's (WHO) Thirteenth General Programme of Work (GPW13) for the period 2018-2025. They were created to translate the health-related Sustainable Development Goals (SDGs) into measurable goals."

    metric = get_contextual_precision_score(
        user_input = input,
        generated_ans = ans,
        expected_output = expected_ans,
        retrieval_context = retrieved_context
    )
    print(f"\nContext Precision metric: {metric.score}")
    print(f"Justification: {metric.reason}\n")

if __name__ == "__main__":
    # summary_score()
    # prompt_alignment_score()
    # hallucination_score()
    # geval_score()
    # conv_geval_score()
    # dag_score()
    # conv_dag()
    # arena_geval()
    # rag_answer_relevancy()
    # rag_faithfulness()
    rag_context_precision()