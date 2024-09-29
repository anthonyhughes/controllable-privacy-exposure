from constants import (
    EVAL_MODELS,
    EVAL_TYPES,
    SUMMARY_TYPES,
)
from evaluate import load
from timeit import default_timer as timer
import argparse

from utils.pii_eval import run_privacy_eval
from utils.reid_eval import run_reidentification_eval
from utils.utility_utils import run_utility_eval

bertscore = load("bertscore")
rouge_eval = load("rouge")


if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", help="Choose a model for inference", default=EVAL_MODELS
    )
    all_types = SUMMARY_TYPES + ["all"]
    parser.add_argument(
        "-t",
        "--tasks",
        help="Choose a model for inference",
        default=SUMMARY_TYPES,
        choices=all_types,
    )
    parser.add_argument(
        "-e",
        "--eval_type",
        help="Choose a type of eval mode",
        default="all",
        choices=EVAL_TYPES,
    )

    args = parser.parse_args()

    models = args.model

    if args.tasks:
        print(f"Target model is {args.tasks}")
        if args.tasks == "all":
            tasks = SUMMARY_TYPES
        else:
            tasks = [args.tasks]

    start = timer()
    for target_model in [models]:
        print(f"Running evaluation pipeline for model: {target_model}")
        print(f"Running eval modes: {args.eval_type}")
        if args.eval_type in ["utility" , "all"]:
            run_utility_eval(target_model=target_model, tasks=tasks)
        if args.eval_type in ["privacy" , "all"]:
            run_privacy_eval(target_model=target_model, tasks=tasks)
        if args.eval_type in ["reidentification", "all"]:            
            run_reidentification_eval(target_model=target_model, tasks=tasks, variation='variation_1')
        end_m = timer() - start
        print(f"Model time to complete in secs: {end_m}")
    end = timer() - start
    print(f"Overall time to complete in secs: {end}")
