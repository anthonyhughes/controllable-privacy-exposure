import argparse
import os
import time
from openai import OpenAI
from constants import (
    BASELINE_SUMMARY_TASK,
    EVAL_MODELS,
    RESULTS_DIR,
    SUMMARY_TYPES,
    PRIV_SUMMARY_TASK,
    IN_CONTEXT_SUMMARY_TASK,
    TASK_SUFFIXES,
)
from mimic.mimic_data import load_original_discharge_summaries, get_ehr_and_summary
from utils.dataset_utils import (
    extract_hadm_ids,
    fetch_example,
    open_legal_data,
    result_file_is_present,
)
from utils.prompts import insert_additional_examples, prompt_prefix_for_task


def inference(client, task, prompt, hadm_id, model):
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
    chat_completion = client.chat.completions.create(
        messages=[prompt],
        model=model,
    )
    summary = chat_completion.choices[0].message.content
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
    model="gpt-4o-mini",
    tasks_suffixes=None,
):
    """Run the OpenAI pipeline"""
    start_time = time.time()
    print("Running the pipelines")
    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    for task in SUMMARY_TYPES[2:]:
        for id in hadm_ids:
            print(f"Running pipeline for {task} on document {id}")
            print("Starting inference")
            # baseline
            baseline_task = f"{task}{BASELINE_SUMMARY_TASK}"
            if BASELINE_SUMMARY_TASK in tasks_suffixes and not result_file_is_present(
                task, id, model
            ):
                baseline_prompt = prompt_prefix_for_task[baseline_task]
                baseline_result = inference(
                    client, task, hadm_id=id, model=model, prompt=baseline_prompt
                )
                save_result(baseline_result, baseline_task, hadm_id=id, model=model)
            # privacy instruct task
            if PRIV_SUMMARY_TASK in tasks_suffixes and not result_file_is_present(
                task, id, model
            ):
                main_prompt = prompt_prefix_for_task[task]
                pseudonymised_result = inference(
                    client, task, hadm_id=id, model=model, prompt=main_prompt
                )
                save_result(pseudonymised_result, task, hadm_id=id, model=model)
            # privacy instruct w/ ICL task
            icl_task = f"{task}{IN_CONTEXT_SUMMARY_TASK}"
            if (
                IN_CONTEXT_SUMMARY_TASK in tasks_suffixes
                and not result_file_is_present(icl_task, id, model)
            ):
                in_context_prompt = prompt_prefix_for_task[icl_task]
                icl_example = fetch_example(task)
                in_context_prompt = in_context_prompt.replace(
                    "[incontext_examples]", icl_example
                )
                in_context_result = inference(
                    client, task, hadm_id=id, model=model, prompt=in_context_prompt
                )
                save_result(in_context_result, icl_task, hadm_id=id, model=model)
            print("Pipeline completed")
    print("All pipelines completed")
    endtime = time.time() - start_time
    print(f"Time taken: {endtime}")


if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--task",
        help="Choose a task for inference",
        default=SUMMARY_TYPES,
        choices=SUMMARY_TYPES,
    )
    parser.add_argument(
        "-m",
        "--model",
        help="Choose a target model for inference",
        default=EVAL_MODELS,
        choices=EVAL_MODELS,
    )
    args = parser.parse_args()

    if args.task:
        print(f"Target task is {args.task}")
        task = args.task

    if task == "legal_court":
        print("Starting legal court inference")
        legal_data = open_legal_data()
        # remove 5 for ICL
        ids = legal_data.keys()
        run(hadm_ids=ids, tasks_suffixes=TASK_SUFFIXES, model=args.model)
    else:
        original_discharge_summaries = load_original_discharge_summaries()
        target_admission_ids = extract_hadm_ids(
            original_discharge_summaries=original_discharge_summaries, n=10000
        )
        # remove the last 5 admission ids
        icl_hadm_ids = target_admission_ids[-1:]
        target_admission_ids = target_admission_ids[:-5]
        run(
            hadm_ids=target_admission_ids[0:100],
            tasks_suffixes=[IN_CONTEXT_SUMMARY_TASK],
            icl_hadm_ids=icl_hadm_ids,
        )
