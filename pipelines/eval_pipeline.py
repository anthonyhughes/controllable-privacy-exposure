from constants import (
    EVAL_MODELS,
    EVAL_TYPES,
    SUMMARY_TYPES,
    TASK_SUFFIXES,
)
from evaluate import load
from timeit import default_timer as timer
import argparse

from utils.pii_eval import run_privacy_eval
from utils.reid_eval import run_reidentification_eval
from utils.reid_eval_v2 import run_reidentification_eval_v2
from utils.utility_utils import run_utility_eval

bertscore = load("bertscore")
rouge_eval = load("rouge")


if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", help="Choose a model for evaluation", default=EVAL_MODELS
    )
    all_types = SUMMARY_TYPES + ["all"]
    parser.add_argument(
        "-t",
        "--tasks",
        help="Choose a model for inference",
        default=SUMMARY_TYPES,
        choices=all_types,
    )
    all_sub_tasks = TASK_SUFFIXES + ["all"]
    parser.add_argument(
        "-st",
        "--sub_tasks",
        help="Choose a sub task for evaluation",
        default=all_sub_tasks,
        choices=all_sub_tasks,
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
        print(f"Target task is {args.tasks}")
        if args.tasks == "all":
            tasks = SUMMARY_TYPES
        else:
            tasks = [args.tasks]

    if args.sub_tasks:
        print(f"Target sub task is {args.sub_tasks}")
        if args.sub_tasks == "all":
            sub_tasks = TASK_SUFFIXES
        else:
            sub_tasks = [args.sub_tasks]

    start = timer()
    for target_model in [models]:
        print(f"Running evaluation pipeline for model: {target_model}")
        print(f"Running eval modes: {args.eval_type}")
        if args.eval_type in ["utility" , "all"]:
            run_utility_eval(target_model=target_model, tasks=tasks, sub_tasks=sub_tasks)
        if args.eval_type in ["privacy" , "all"]:
            run_privacy_eval(target_model=target_model, tasks=tasks, sub_tasks=sub_tasks)
        if args.eval_type in ["reidentification", "all"]:            
            # run_reidentification_eval(target_model=target_model, tasks=tasks, variation='variation_1', sub_tasks=sub_tasks)            
            run_reidentification_eval_v2(target_privacy_file="mistral-7b-instruct-v0.3-bnb-4bit-2024-10-03-20-56-01.json")
        end_m = timer() - start
        print(f"Model time to complete in secs: {end_m}")
    end = timer() - start
    print(f"Overall time to complete in secs: {end}")
