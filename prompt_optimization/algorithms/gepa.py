"""
The Genetic-Pareto (GEPA) method is a prompt optimization algorithm that combines evolutionary optimization with multi-objective Pareto selection
to improve prompts while keeping them diverse enough for various problems.

Let's suppose we have an agent that needs to generate code while also generating through creative writing the corresponding documentation.
Instead of coming with one 'best' prompt that excels in one particular task but lacks in another (e.g., it could be better for generating code but not that good at writing),
GEPA founds a 'middle' prompt that's good enough at both tasks at the same time (this is called Pareto optimal).

GEPA samples trajectories (e.g., reasoning, tool calls, and tool outputs) and reflects on them in natural language to diagnose problems, propose and test
prompt updates, and combine complementary lessons from the Pareto frontier of its own attempts. 

Additional information found in https://deepeval.com/docs/prompt-optimization-gepa
You can read GEPA's paper in https://arxiv.org/pdf/2507.19457
"""

from deepeval.optimizer import PromptOptimizer
from deepeval.optimizer.algorithms import GEPA
from deepeval.optimizer.policies import TieBreaker

def gepa_optimizer(model_callback):
    optimizer = PromptOptimizer(
        algorithm=GEPA(
            iterations=10,  # Total number of mutation attempts. Defaults to 5
            pareto_size=5,  # Number of goldens in the Pareto validation set. Defaults to 3
            minibatch_size=4,   # Number of goldens drawn for feedback per iteration. Automatically clamped to available data. Defaults to 8
            random_seed=42, # Controls randomness in golden splitting, minibatch sampling, Pareto selection, and tie-breaking. Defaults to time.time_ns()
            tie_breaker=TieBreaker.PREFER_CHILD    # Policy for breaking ties (PREFER_ROOT, PREFER_CHILD, RANDOM). Defaults to PREFER_CHILD
        ),
        model_callback=model_callback
    )

    return optimizer