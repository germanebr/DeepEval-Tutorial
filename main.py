from metrics.summarization import get_summary_score
from metrics.prompt_alignment import get_prompt_alignment_score

from models.gcp_gemini import GCP_GENERATION_MODEL

def summary_metric():
    print("---  Summarization Score  ---")

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
    with open("./prompts/summarization_prompt.md") as f:
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
    with open("./prompts/prompt_alignment_prompt.md") as f:
        prompt_alignment_prompt = f.read()

    ans = GCP_GENERATION_MODEL().generate(prompt_alignment_prompt, input)
    print(f"Generated answer:\n{ans}")

    # Obtain the metric
    metric = get_prompt_alignment_score(prompt_alignment_prompt, input, ans)
    print(f"\nPrompt alignment metric: {metric.score}")
    print(f"Justification: {metric.reason}\n")

if __name__ == "__main__":
    summary = summary_metric()
    prompt_alignment = prompt_alignment_score()