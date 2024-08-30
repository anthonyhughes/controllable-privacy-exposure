import os
from constants import (
    BASELINE_SUMMARY_TASK,
    IN_CONTEXT_SUMMARY_TASK,
    PRIV_SUMMARY_TASK,
    RESULTS_DIR,
)
from utils.dataset_utils import result_file_is_present
from utils.prompts import insert_additional_examples
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
    icl_hadm_ids=[],
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
        save_result(baseline_result, f"{task}_baseline", hadm_id=id, model=model)
        time.sleep(10)
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
        time.sleep(10)
    # privacy instruct w/ ICL task
    icl_task = f"{task}{IN_CONTEXT_SUMMARY_TASK}"
    if IN_CONTEXT_SUMMARY_TASK in tasks_suffixes and not result_file_is_present(
        icl_task, id, model
    ):
        print("Running private inference w\ ICL")
        in_context_prompt = prompt_prefix_for_task[icl_task]
        in_context_prompt = insert_additional_examples(
            task, in_context_prompt, icl_hadm_ids
        )
        in_context_result = inference_fnc(
            client, task, hadm_id=id, model=model, prompt=in_context_prompt
        )
        save_result(in_context_result, f"{task}_in_context", hadm_id=id, model=model)
        time.sleep(10)
