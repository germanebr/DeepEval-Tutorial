"""
The Multiprompt Instruction PRoposal Optimizer Version 2 (MIPROv2) method is a prompt optimization algorithm that combines instruction proposal with few-shot
demonstration and Bayesian Optimization to find the optimal prompt configuration. This method uses systematic search to obtain the best set of instructions
and examples for the given task.

Additional information found in https://deepeval.com/docs/prompt-optimization-miprov2
You can read MIPRO's paper in https://arxiv.org/pdf/2406.11695
"""

from deepeval.optimizer import PromptOptimizer as DeepEvalPromptOptimizer
from deepeval.optimizer.algorithms import MIPROV2

from models.gcp_gemini import gcp_gemini_eval_model

def mipro_optimizer(model_callback, metrics:list):
    optimizer = DeepEvalPromptOptimizer(
        algorithm=MIPROV2(
            num_candidates=10,  # Number of diverse instruction candidates to generate in the proposal phase. Defaults to 10
            num_trials=20,  # Number of Bayesian Optimization trials to run. Each trial evaluates a different combination of instruction,demo_set. Defaults to 20
            minibatch_size=25,  # Number of goldens sampled per trial for evaluation. Defaults to 25
            minibatch_full_eval_steps=10,   # Run a full evaluation on all goldens every N trials. Defaults to 10
            max_bootstrapped_demos=4,   # Maximum number of model-generated outputs that passed validation per demo set. Defaults to 4
            max_labeled_demos=4,    # Max number of labeled demonstrations per demo set. Defaults to 4
            num_demo_sets=5,    # Number of different demo set configurations to create. More sets = more variety to optimize. Defaults to 5
            random_seed=42, # Controls randomness in candidate generation, demo bootstrapping, and trial sampling. Defaults to time.time_ns()
        ),
        model_callback=model_callback,
        metrics=metrics,
        optimizer_model=gcp_gemini_eval_model
    )

    return optimizer