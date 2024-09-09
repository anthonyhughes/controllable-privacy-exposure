from constants import (
    EVAL_MODELS,
    SUMMARY_TYPES,
)
from evaluate import load

from utils.pii_eval import run_privacy_eval
from timeit import default_timer as timer
import argparse

from utils.utility_utils import run_utility_eval, print_results_to_latex

bertscore = load("bertscore")
rouge_eval = load("rouge")


if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", help="Choose a model for inference", default=EVAL_MODELS
    )
    parser.add_argument(
        "-t",
        "--tasks",
        help="Choose a model for inference",
        default=SUMMARY_TYPES,
        choices=SUMMARY_TYPES,
    )
    args = parser.parse_args()

    models = args.model

    if args.tasks:
        print(f"Target model is {args.tasks}")
        tasks = [args.tasks]

    start = timer()
    for target_model in [models]:
        print(f"Running evaluation pipeline for model: {target_model}")
        run_utility_eval(target_model=target_model, tasks=tasks)
        run_privacy_eval(target_model=target_model, tasks=tasks)
        end_m = timer() - start
        print(f"Model time to complete in secs: {end_m}")
    end = timer() - start
    print(f"Overall time to complete in secs: {end}")
