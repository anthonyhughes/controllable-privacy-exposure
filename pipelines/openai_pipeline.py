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
    extract_hadm_ids_from_dir,
    fetch_example,
    open_legal_data,
    result_file_is_present,
)
from utils.inference import all_inference_tasks
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
    tasks=SUMMARY_TYPES,
    task_suffixes=TASK_SUFFIXES
):
    start_time = time.time()
    print("Running the openai pipeline")

    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    for task in tasks:
        for i, id in enumerate(hadm_ids):
            print(
                f"Running pipeline for {task} on document {id} - {i+1}/{len(hadm_ids)}"
            )
            if id in ["28664981", "21441082"]:
                continue
            all_inference_tasks(
                id,
                task,
                prompt_prefix_for_task,
                inference_fnc=inference,
                client=client,
                tasks_suffixes=task_suffixes,
                model=model,
                sleep=5,
            )            
            print(f"Pipeline completed - {id}")
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
        target_admission_ids = extract_hadm_ids_from_dir(
            "llama-3-8b-Instruct-bnb-4bit", "brief_hospital_course"
        )
        run(
            hadm_ids=target_admission_ids,
            tasks=[task],
        )
