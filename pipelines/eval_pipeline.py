from constants import (
    EVAL_MODELS,
)
from evaluate import load

from utils.pii_eval import run_privacy_eval
from timeit import default_timer as timer
import argparse

from utils.utility_utils import run_utility_eval

bertscore = load("bertscore")
rouge_eval = load("rouge")


if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        help="Choose a model for inference",
        default=EVAL_MODELS,
        choices=EVAL_MODELS,
    )
    args = parser.parse_args()

    if args.model:
        print(f"Target model is {args.model}")
        models = [args.model]

    start = timer()
    for target_model in models:
        print(f"Running evaluation pipeline for model: {target_model}")
        run_utility_eval(target_model=target_model)
        run_privacy_eval(target_model=target_model)
    end = timer() - start
    print(f"Time to complete in secs: {end}")
