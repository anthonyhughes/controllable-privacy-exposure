import argparse
import json
import os
import time
from openai import OpenAI
from constants import (
    BASELINE_SUMMARY_TASK,
    BATCH_JOBS_DIR,
    EVAL_MODELS,
    IN_CONTEXT_SUMMARY_TASK,
    PRIV_SUMMARY_TASK,
    RESULTS_DIR,
    SUMMARY_TYPES,
    TASK_SUFFIXES,
    BATCH_FLAGS,
)
from mimic.mimic_data import get_ehr_and_summary
from utils.dataset_utils import (
    extract_hadm_ids_from_dir,
    fetch_example,
    open_cnn_data,
    open_legal_data,
    result_file_is_present,
)
from utils.prompts import prompt_prefix_for_task


def build_prompt_for_task(
    task,
    prompt,
    hadm_id,
):
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
    return prompt


def marshall_prompt_into_openai_object(message, model, job_id):
    object = {
        "custom_id": job_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": model, "messages": [message]},
    }
    return object


def build_all_variations_for_task(
    id,
    task,
    tasks_suffixes=TASK_SUFFIXES,
    model="gpt-4o-mini",
    cdatetime="now",
):
    print("Updating batch job for {task} on document {id}")
    # baseline
    baseline_task = f"{task}{BASELINE_SUMMARY_TASK}"
    if BASELINE_SUMMARY_TASK in tasks_suffixes and not result_file_is_present(
        task, id, model
    ):
        print("Adding baseline task to batch")
        baseline_prompt = prompt_prefix_for_task[baseline_task]
        baseline_prompt = build_prompt_for_task(task, baseline_prompt, id)
        baseline_job = marshall_prompt_into_openai_object(
            baseline_prompt, model, f"{baseline_task}_{id}"
        )
        append_result_to_batch_job(baseline_job, model, cdatetime)
    else:
        print(f"Skipping document {id} on task {task}")

    # privacy instruct task
    if PRIV_SUMMARY_TASK in tasks_suffixes and not result_file_is_present(
        task, id, model
    ):
        print("Adding private task to batch")
        main_prompt = prompt_prefix_for_task[task]
        main_prompt = build_prompt_for_task(task, main_prompt, id)
        main_job = marshall_prompt_into_openai_object(
            main_prompt, model, f"{task}_{id}"
        )
        append_result_to_batch_job(main_job, model, cdatetime)
    else:
        print(f"Skipping document {id} on task {task}")

        # save_result_to_batch(pseudonymised_result, task, hadm_id=id, model=model)
    # privacy instruct w/ ICL task
    icl_task = f"{task}{IN_CONTEXT_SUMMARY_TASK}"
    if IN_CONTEXT_SUMMARY_TASK in tasks_suffixes and not result_file_is_present(
        icl_task, id, model
    ):
        print("Adding one-shot private task to batch")
        in_context_prompt = prompt_prefix_for_task[icl_task]
        icl_example = fetch_example(task)
        in_context_prompt = in_context_prompt.replace(
            "[incontext_examples]", icl_example
        )
        in_context_prompt = build_prompt_for_task(task, in_context_prompt, id)
        baseline_job = marshall_prompt_into_openai_object(
            in_context_prompt, model, f"{icl_task}_{id}"
        )
        append_result_to_batch_job(baseline_job, model, cdatetime)
    else:
        print(f"Skipping document {id} on task {task}")


def append_result_to_batch_job(openai_job, model, cdatetime):
    """Save result"""
    if not os.path.exists(f"{BATCH_JOBS_DIR}"):
        os.makedirs(f"{BATCH_JOBS_DIR}")
    with open(
        f"{BATCH_JOBS_DIR}/{model}-{cdatetime}.jsonl",
        "a",
    ) as f:
        serialized = json.dumps(openai_job)
        f.write(f"{serialized}\n")


def prep_batch_job(cdatetime, client, model):
    target_batch = f"{BATCH_JOBS_DIR}/{model}-{cdatetime}.jsonl"
    batch_file = client.files.create(file=open(target_batch, "rb"), purpose="batch")
    details_to_save = {
        "id": batch_file.id,
        "filename": batch_file.filename,
        "created_at": batch_file.created_at,
    }
    with open(f"{target_batch}.batch", "w") as f:
        json.dump(details_to_save, f)
    return batch_file


def run(hadm_ids, model="gpt-4o-mini", tasks=SUMMARY_TYPES, start_time="now"):
    start_time = time.time()
    print("Running the openai pipeline")

    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    print("Building all jobs")
    for task in tasks:
        for i, id in enumerate(hadm_ids):
            print(f"Create batch for OpenAI")
            print(f"{task} on document {id} - {i+1}/{len(hadm_ids)}")
            build_all_variations_for_task(id, task, model=model, cdatetime=start_time)
            print(f"Pipeline completed - {id}")

    batch_file = prep_batch_job(cdatetime=start_time, client=client, model=model)
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(batch_job)

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
    parser.add_argument(
        "-f",
        "--flag",
        help="Choose a batch mode",
        default=BATCH_FLAGS[0],
        choices=BATCH_FLAGS,
    )
    parser.add_argument(
        "-j",
        "--job_id",
        help="Choose a job identifier to check upon"
    )
    parser.add_argument(
        "-fi",
        "--file_id",
        help="Choose an output file identifier"
    )
    args = parser.parse_args()

    if args.task:
        print(f"Target task is {args.task}")
        task = args.task

    if args.flag:
        flag = args.flag
        print(f"Flag {flag}")

    if args.model:
        model = args.model
        print(f"Target model: {model}")

    if flag == "batch":
        if task == "legal_court":
            print("Starting legal court inference")
            legal_data = open_legal_data()
            # remove 5 for ICL
            ids = legal_data.keys()
            run(hadm_ids=ids, tasks=[task], model=args.model)
        elif task == "cnn":
            print("Starting CNN inference")
            news_data = open_cnn_data()
            # remove 5 for ICL
            ids = list(news_data.keys())
            ids = ids[:-5]
            run(hadm_ids=ids, tasks=[task], model=args.model)
        else:
            target_admission_ids = extract_hadm_ids_from_dir(
                "llama-3-8b-Instruct-bnb-4bit", "brief_hospital_course"
            )
            run(hadm_ids=target_admission_ids, tasks=[task], model=args.model)
    elif flag == "check":
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        job_id = args.job_id
        batch_job = client.batches.retrieve(job_id)
        print(batch_job)
    elif flag == "retrieve":
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        file_id = args.file_id
        result = client.files.content(file_id).content
        result_file_name = f"{RESULTS_DIR}/{model}/openai_batch_results.jsonl"

        with open(result_file_name, 'wb') as file:
            file.write(result)

