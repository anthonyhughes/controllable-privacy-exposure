import os
from constants import (
    BASELINE_SUMMARY_TASK,
    IN_CONTEXT_SUMMARY_TASK,
    PRIV_SUMMARY_TASK,
    RESULTS_DIR,
)
from utils.dataset_utils import fetch_example, result_file_is_present
import time


def save_result(result, task, hadm_id, model):
    """Save result"""
    if not os.path.exists(f"data/results/{model}/{task}"):
        os.makedirs(f"data/results/{model}/{task}")
    with open(
        f"{RESULTS_DIR}/{model}/{task}/{hadm_id}-discharge-inputs.txt",
        "w",
    ) as f:
        f.write(result)


def all_inference_tasks(
    id,
    task,
    prompt_prefix_for_task,
    inference_fnc,
    client,
    tasks_suffixes=[],
    model="gpt-4o-mini",
    sleep=0,
):
    print(f"Running pipeline for {task} on patient {id}")
    print("Starting inference")
    # baseline
    baseline_task = f"{task}{BASELINE_SUMMARY_TASK}"
    if BASELINE_SUMMARY_TASK in tasks_suffixes and not result_file_is_present(
        task, id, model
    ):
        print("Running baseline inference")
        baseline_prompt = prompt_prefix_for_task[baseline_task]
        baseline_result = inference_fnc(
            client, task, hadm_id=id, model=model, prompt=baseline_prompt
        )
        save_result(baseline_result, baseline_task, hadm_id=id, model=model)
        time.sleep(sleep)
    # privacy instruct task
    if PRIV_SUMMARY_TASK in tasks_suffixes and not result_file_is_present(
        task, id, model
    ):
        print("Running private inference")
        main_prompt = prompt_prefix_for_task[task]
        pseudonymised_result = inference_fnc(
            client, task, hadm_id=id, model=model, prompt=main_prompt
        )
        save_result(pseudonymised_result, task, hadm_id=id, model=model)
        time.sleep(sleep)
    # privacy instruct w/ ICL task
    icl_task = f"{task}{IN_CONTEXT_SUMMARY_TASK}"
    if IN_CONTEXT_SUMMARY_TASK in tasks_suffixes and not result_file_is_present(
        icl_task, id, model
    ):
        print("Running private inference w\ ICL")
        in_context_prompt = prompt_prefix_for_task[icl_task]
        icl_example = fetch_example(task)
        in_context_prompt = in_context_prompt.replace(
            "[incontext_examples]", icl_example
        )
        in_context_result = inference_fnc(
            client, task, hadm_id=id, model=model, prompt=in_context_prompt
        )
        save_result(in_context_result, icl_task, hadm_id=id, model=model)
        time.sleep(sleep)
