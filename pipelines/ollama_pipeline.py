import argparse
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
    extract_hadm_ids_from_dir,
    fetch_example,
    result_file_is_present,
)
from utils.prompts import prompt_prefix_for_task


def inference(task, prompt, hadm_id, model):
    """Run the openai query"""
    ehr, _ = get_ehr_and_summary(task, hadm_id)
    prompt = f"""### Instruction:
                {prompt}

                ### Input:
                {ehr}

                ### Response:"""
    response = ollama.generate(
        model=model,
        prompt=prompt,
    )
    summary = response["response"]
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
    task=None,
):
    """Run the OpenAI pipeline"""
    start_time = time.time()
    print("Running the pipelines")
    for i, id in enumerate(hadm_ids):
        print(f"Running pipeline for {task} on patient {id} - {i+1}/{len(hadm_ids)}")
        print("Starting inference")
        # baseline
        baseline_task = f"{task}{BASELINE_SUMMARY_TASK}"
        if BASELINE_SUMMARY_TASK in tasks_suffixes and not result_file_is_present(
            task, id, model
        ):
            print('Starting baseline')
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
            print('Starting priv')
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
            print('Starting ICL')
            in_context_prompt = prompt_prefix_for_task[icl_task]
            icl_example = fetch_example(task)
            in_context_prompt = in_context_prompt.replace("[incontext_examples]", icl_example)
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
        # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task",help="Choose a task for inference", default=SUMMARY_TYPES, choices=SUMMARY_TYPES)
    args = parser.parse_args()

    if args.task:
        print(f"Target task is {args.task}")
        task = args.task
        
    if task == "legal_court":
        print("Starting legal court inference")
        pass
    else:
        original_discharge_summaries = load_original_discharge_summaries()
        target_admission_ids = extract_hadm_ids(
            original_discharge_summaries=original_discharge_summaries, n=0
        )
        # remove the last 5 admission ids
        icl_hadm_ids = target_admission_ids[-1:]
        target_admission_ids = extract_hadm_ids_from_dir('llama-3-8b-Instruct-bnb-4bit', 'brief_hospital_course')
        run(
            hadm_ids=target_admission_ids,
            tasks_suffixes=TASK_SUFFIXES,
            task=task,
            model="llama3.1:70b"
        )
