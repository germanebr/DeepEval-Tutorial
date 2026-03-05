from metrics.summarization import get_summary_score

from models.gcp_gemini import GCP_GENERATION_MODEL

def summarize():
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
    print(f"Justification: {metric.reason}")

if __name__ == "__main__":
    summary = summarize()