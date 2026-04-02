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
from rag.metrics.contextual_relevancy import get_contextual_relevancy_score

from models.gcp_gemini import gcp_gemini_eval_model
from models.custom_prompt_optimizer import MyPromptOptimizer

from deepeval.test_case import LLMTestCase, LLMTestCaseParams, Turn
from deepeval.metrics.g_eval import Rubric
from deepeval.metrics import SummarizationMetric

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

def rag_context_recall():
    print("--- Contextual Recall ---")

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

    metric = get_contextual_recall_score(
        user_input = input,
        generated_ans = ans,
        expected_output = expected_ans,
        retrieval_context = retrieved_context
    )
    print(f"\nContext Recall metric: {metric.score}")
    print(f"Justification: {metric.reason}\n")

def rag_context_relevancy():
    print("--- Contextual Relevancy ---")

    input = "What are the Triple Billion targets?"
    print(input)

    with open("./docs/rag_context_sample.json", "r", encoding="utf-8") as f:
        context = json.load(f)
        str_context = json.dumps(context)

        retrieved_context = [i['section_content'] for i in context]

    ans = GCP_GENERATION_MODEL().generate(input, str_context)
    print(ans)

    metric = get_contextual_relevancy_score(
        user_input = input,
        generated_ans = ans,
        retrieval_context = retrieved_context
    )
    print(f"\nContext Relevancy metric: {metric.score}")
    print(f"Justification: {metric.reason}\n")

def prompt_optimize():
    print("--- Prompt Optimization ---")

    # Get the prompt
    with open("./custom/prompts/summarization_prompt.md") as f:
        prompt = f.read()

    input = """
    The 'coverage score' is calculated as the percentage of assessment questions
    for which both the summary and the original document provide a 'yes' answer. This
    method ensures that the summary not only includes key information from the original
    text but also accurately represents it. A higher coverage score indicates a
    more comprehensive and faithful summary, signifying that the summary effectively
    encapsulates the crucial points and details from the original content.
    """

    optimizer = MyPromptOptimizer(
        prompt=prompt,
        goldens=[
            # Golden Case 1: Use the existing input, but provide a hand-crafted, ideal summary
            LLMTestCase(
                input=input,
                actual_output="The coverage score is a percentage of assessment questions where both the summary and original document provide a 'yes' answer. This method ensures the summary includes and accurately represents key information, with a higher score indicating a more comprehensive and faithful summary."
            ),
            # Golden Case 2: Add a second, distinct example with a hand-crafted ideal summary
            # This example is taken from the "3. Progress in achieving the Triple Billion targets" section of your rag_context_sample.json
            LLMTestCase(
                input="The Triple Billion targets, a cornerstone of WHO's Thirteenth General Programme of Work (GPW13), track the collective progress by WHO and its Member States in promoting, providing and protecting health worldwide. The latest update illustrates the progress made since the start of GPW13 in 2018 and the key challenges particularly in the universal health coverage and health emergencies protection billions. Anchored in the health-related SDGS, WHO's GPW13 provides a strategic roadmap to enhance health and well-being for all (1). More importantly, GPW13 championed the results framework with impact measurement at its core to reinforce organizational accountability and transparency. The foundation of the impact measurement is the 46 outcome indicators centred on promoting the overall health and quality of life for the people, providing essential health services, and protecting people from health emergencies. The Triple Billion targets give strategic clarity to the impact measurement of GPW13. By summarizing the outcome indicators into three ambitious, but easy to understand and communicate goals, the Triple Billion targets effectively convey global health priorities, motivate collective action by the global community, facilitate transparency and accountability, and complement the strategic priorities set by WHO's GPW13 to address current and emerging global health priorities. For GPW13, the WHO Member States aim to enable one billion more people achieve better health and overall well-being, to provide one billion more people with access to essential health services without incurring financial hardship, and to better protect one billion more people from health emergencies worldwide by the end of the GPW13, extended from 2023 to 2025. This chapter provides a summary of the progress made in achieving the Triple Billion targets with projections to 2025, incorporating the latest available data from the outcome indicators at the country level.",
                actual_output="The WHO's Triple Billion targets, part of its GPW13 program, track global health progress in promoting, providing, and protecting health. These targets, based on 46 outcome indicators, aim for one billion more people to achieve better health, access essential services without financial hardship, and be protected from health emergencies by 2025. They provide strategic clarity, motivate action, and ensure accountability for global health priorities."
            )
        ],
        metrics=[
            SummarizationMetric(
                threshold=0.7,  # Adjust threshold as needed
                model=gcp_gemini_eval_model,
                assessment_questions=[
                    "Does the summary accurately reflect the key information in the input?",
                    "Does the summary cover all important aspects of the input?",
                    "Is the summary free from information not present in the input?"
                ]
            )
        ]
    )

    print("\t> Optimizing through GEPA")
    gepa_prompt = optimizer.optimize_prompt(algorithm="gepa")
    print(f"\t{gepa_prompt}")

    print("\t> Optimizing through MIPROv2")
    mipro_prompt = optimizer.optimize_prompt(algorithm="mipro")
    print(f"\t{mipro_prompt}")

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
    # rag_context_precision()
    # rag_context_recall()
    # rag_context_relevancy()
    prompt_optimize()