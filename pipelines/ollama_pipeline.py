import os
import time
import ollama
from constants import (
    BASELINE_SUMMARY_TASK,
    RESULTS_DIR,
    SUMMARY_TYPES,
    PRIV_SUMMARY_TASK,
    IN_CONTEXT_SUMMARY_TASK,
    TASK_SUFFIXES,
)
from mimic.mimic_data import load_original_discharge_summaries, get_ehr_and_summary
from utils.dataset_utils import (
    extract_hadm_ids,
    result_file_is_present,
)
from utils.prompts import insert_additional_examples, prompt_prefix_for_task


def inference(task, prompt, hadm_id, model):
    """Run the openai query"""
    ehr, _ = get_ehr_and_summary(task, hadm_id)
    prompt = {
        "role": "user",
        "content": f"""
                ### Instruction:
                {prompt}

                ### Input:
                {ehr}

                ### Response:
            """,
    }
    response = ollama.chat(
        model=model,
        messages=[
            prompt,
        ],
    )
    summary = response["message"]["content"]
    return summary


def save_result(openai_result, task, hadm_id, model):
    """Save result"""
    if not os.path.exists(f"data/results/{model}/{task}"):
        os.makedirs(f"data/results/{model}/{task}")
    with open(
        f"{RESULTS_DIR}/{model}/{task}/{hadm_id}-discharge-inputs.txt",
        "w",
    ) as f:
        f.write(openai_result)


def run(
    hadm_ids,
    model="llama3.1:70b",
    tasks_suffixes=None,
    icl_hadm_ids=None,
):
    """Run the OpenAI pipeline"""
    start_time = time.time()
    print("Running the pipelines")
    for task in SUMMARY_TYPES:
        for id in hadm_ids:
            print(f"Running pipeline for {task} on patient {id}")
            print("Starting inference")
            # baseline
            baseline_task = f"{task}{BASELINE_SUMMARY_TASK}"
            if BASELINE_SUMMARY_TASK in tasks_suffixes and not result_file_is_present(
                task, id, model
            ):
                baseline_prompt = prompt_prefix_for_task[baseline_task]
                baseline_result = inference(
                    task, hadm_id=id, model=model, prompt=baseline_prompt
                )
                save_result(
                    baseline_result, f"{task}_baseline", hadm_id=id, model=model
                )
            # privacy instruct task
            if PRIV_SUMMARY_TASK in tasks_suffixes and not result_file_is_present(
                task, id, model
            ):
                main_prompt = prompt_prefix_for_task[task]
                pseudonymised_result = inference(
                    task, hadm_id=id, model=model, prompt=main_prompt
                )
                save_result(pseudonymised_result, task, hadm_id=id, model=model)
            # privacy instruct w/ ICL task
            icl_task = f"{task}{IN_CONTEXT_SUMMARY_TASK}"
            if (
                IN_CONTEXT_SUMMARY_TASK in tasks_suffixes
                and not result_file_is_present(icl_task, id, model)
            ):
                in_context_prompt = prompt_prefix_for_task[icl_task]
                in_context_prompt = insert_additional_examples(
                    task, in_context_prompt, icl_hadm_ids
                )
                in_context_result = inference(
                    task, hadm_id=id, model=model, prompt=in_context_prompt
                )
                save_result(
                    in_context_result, f"{task}_in_context", hadm_id=id, model=model
                )
            print("Pipeline completed")
    print("All pipelines completed")
    endtime = time.time() - start_time
    print(f"Time taken: {endtime}")


if __name__ == "__main__":
    original_discharge_summaries = load_original_discharge_summaries()
    target_admission_ids = extract_hadm_ids(
        original_discharge_summaries=original_discharge_summaries, n=0
    )
    # remove the last 5 admission ids
    icl_hadm_ids = target_admission_ids[-1:]
    
    run(
        hadm_ids=target_admission_ids,
        tasks_suffixes=TASK_SUFFIXES,
        icl_hadm_ids=icl_hadm_ids,
        model="llama3.1:70b"
    )
