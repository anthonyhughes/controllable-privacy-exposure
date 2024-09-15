import argparse
import os
from anthropic import Anthropic
import time
from constants import (
    EVAL_MODELS,
    SUMMARY_TYPES,
    TASK_SUFFIXES,
)
from mimic.mimic_data import get_ehr_and_summary
from utils.dataset_utils import (
    extract_hadm_ids_from_dir,
    open_cnn_data,
    open_legal_data,
)
from utils.inference import all_inference_tasks
from utils.prompts import prompt_prefix_for_task


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
    chat_completion = client.messages.create(
        max_tokens=2048,
        messages=[prompt],
        model=model,
    )
    summary = chat_completion.content
    return summary[0].text


def run(hadm_ids, model="claude-3-5-sonnet-20240620", tasks_suffixes=None, tasks=[]):
    print("Running the claude pipeline")

    client = Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )

    for task in tasks:
        for i, id in enumerate(hadm_ids):
            print(
                f"Running pipeline for {task} on document {id} - {i+1}/{len(hadm_ids)}"
            )
            all_inference_tasks(
                id,
                task,
                prompt_prefix_for_task,
                inference_fnc=inference,
                client=client,
                tasks_suffixes=tasks_suffixes,
                model=model,
                sleep=20,
            )
            print(f"Pipeline completed - {id}")
    print("All pipelines completed")


if __name__ == "__main__":
    start_time = time.time()
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
    elif task == "cnn":
        print("Starting CNN inference")
        news_data = open_cnn_data()
        # remove 5 for ICL
        ids = news_data.keys()
        run(
            hadm_ids=ids,
            model=args.model,
            tasks_suffixes=TASK_SUFFIXES,
            tasks=[task],
        )
    else:
        target_admission_ids = extract_hadm_ids_from_dir(
            "llama-3-8b-Instruct-bnb-4bit", "brief_hospital_course"
        )
        run(
            hadm_ids=target_admission_ids,
            model="claude-3-5-sonnet-20240620",
            tasks_suffixes=TASK_SUFFIXES,
            tasks=[task],
        )
    endtime = time.time() - start_time
    print(f"Time taken: {endtime}")
