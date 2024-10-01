import argparse
import json
import os
import time
from openai import OpenAI
from constants import (
    BASELINE_SUMMARY_TASK,
    BATCH_JOBS_DIR,
    BATCH_RESULTS_DIR,
    EVAL_MODELS,
    IN_CONTEXT_SUMMARY_TASK,
    PRIV_SUMMARY_TASK,
    RESULTS_DIR,
    SANI_SUMM_SUMMARY_TASK,
    SANITIZE_TASK,
    SUMMARY_TYPES,
    TASK_SUFFIXES,
    BATCH_FLAGS,
    MAX_TOKENS,
)
from mimic.mimic_data import get_ehr_and_summary
from utils.dataset_utils import (
    extract_hadm_ids_from_dir,
    fetch_example,
    open_cnn_data,
    open_generated_summary,
    open_legal_data,
    result_file_is_present,
)
from utils.prompt_variations import variations, sanitize_prompt


def build_instruction_prompt_with_document(
    instruction,
    document,
):
    return {
        "role": "user",
        "content": f"""
                ### Instruction:
                {instruction}

                ### Input:
                {document}

                ### Response:
            """,
    }


def build_prompt_for_task(
    task,
    prompt,
    hadm_id,
):
    ehr, _ = get_ehr_and_summary(task, hadm_id)
    prompt = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"""
                    ### Instruction:
                    {prompt}

                    ### Input:
                    {ehr}

                    ### Response:
            """,
            }
        ],
    }
    return prompt


def marshall_prompt_into_openai_object(message, model, job_id):
    object = {
        "recordId": job_id,
        "modelInput": {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": MAX_TOKENS,
            "messages": [message],
        },
    }
    return object


def build_all_variations_for_task(
    id,
    task,
    tasks_suffixes=TASK_SUFFIXES,
    model="",
    cdatetime="now",
):
    print(f"Updating batch job for {task} on document {id}")
    # baseline
    for prompt_prefix_for_task in variations:
        v_name = prompt_prefix_for_task["name"]
        print(
            f"Running inference for summary type : {task} and variation: {v_name} on model: {model}"
        )
        baseline_task = f"{task}{BASELINE_SUMMARY_TASK}"
        if BASELINE_SUMMARY_TASK in tasks_suffixes and not result_file_is_present(
            task, id, model, v_name
        ):
            print("Adding baseline task to batch")
            baseline_prompt = prompt_prefix_for_task[baseline_task]
            baseline_prompt = build_prompt_for_task(task, baseline_prompt, id)
            baseline_job = marshall_prompt_into_openai_object(
                baseline_prompt, model, f"{v_name}_{baseline_task}_{id}"
            )
            append_result_to_batch_job(baseline_job, model, cdatetime)
        else:
            print(f"Skipping document {id} on task {baseline_task}")

        # privacy instruct task
        if PRIV_SUMMARY_TASK in tasks_suffixes and not result_file_is_present(
            task, id, model, v_name
        ):
            print("Adding private task to batch")
            main_prompt = prompt_prefix_for_task[task]
            main_prompt = build_prompt_for_task(task, main_prompt, id)
            main_job = marshall_prompt_into_openai_object(
                main_prompt, model, f"{v_name}_{task}_{id}"
            )
            append_result_to_batch_job(main_job, model, cdatetime)
        else:
            print(f"Skipping document {id} on task {task}")

        # privacy instruct w/ ICL task
        icl_task = f"{task}{IN_CONTEXT_SUMMARY_TASK}"
        if IN_CONTEXT_SUMMARY_TASK in tasks_suffixes and not result_file_is_present(
            icl_task, id, model, v_name
        ):
            print("Adding one-shot private task to batch")
            in_context_prompt = prompt_prefix_for_task[icl_task]
            icl_example = fetch_example(task)
            in_context_prompt = in_context_prompt.replace(
                "[incontext_examples]", icl_example
            )
            in_context_prompt = build_prompt_for_task(task, in_context_prompt, id)
            icl_job = marshall_prompt_into_openai_object(
                in_context_prompt, model, f"{v_name}_{icl_task}_{id}"
            )
            append_result_to_batch_job(icl_job, model, cdatetime)
        else:
            print(f"Skipping document {id} on task {icl_task}")

        # sanitize
        sani_task = f"{task}{SANITIZE_TASK}"
        # if sanitisation has not been ran
        if (
            SANITIZE_TASK in tasks_suffixes
            and not result_file_is_present(sani_task, id, model, v_name)
            and v_name == "variation_1"
        ):
            print("Adding sani task to batch")
            sani_prompt = build_prompt_for_task(task, sanitize_prompt, id)
            job_to_store = marshall_prompt_into_openai_object(
                sani_prompt, model, f"{v_name}_{sani_task}_{id}"
            )
            append_result_to_batch_job(job_to_store, model, cdatetime)
        else:
            print(f"Skipping document {id} on task {sani_task}")

        # now summarize
        sani_summ_task = f"{task}{SANI_SUMM_SUMMARY_TASK}"
        # check summary not available and sanitized doc is available
        if (
            SANI_SUMM_SUMMARY_TASK in tasks_suffixes
            and not result_file_is_present(sani_summ_task, id, model, v_name)
            and result_file_is_present(sani_task, id, model, v_name)
        ):
            print("Adding sani summ task to batch")
            sanitized_doc = open_generated_summary(
                task=sani_task, hadm_id=id, model=model, variation=v_name
            )
            sani_summ_prompt = prompt_prefix_for_task[sani_summ_task]
            sani_summ_prompt = build_instruction_prompt_with_document(
                sani_summ_prompt, sanitized_doc
            )
            job_to_store = marshall_prompt_into_openai_object(
                sani_summ_prompt, model, f"{v_name}_{sani_summ_task}_{id}"
            )
            append_result_to_batch_job(job_to_store, model, cdatetime)
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


def save_batch_job(model, cdatetime, details_to_save):
    target_batch = f"{BATCH_JOBS_DIR}/{model}-{cdatetime}.jsonl"
    with open(f"{target_batch}.batch", "w") as f:
        json.dump(details_to_save, f)


def create_batch(hadm_ids, model, tasks=SUMMARY_TYPES, start_time="now"):
    start_time = time.time()
    print("Running the aws bedrock pipeline")

    print("Building all jobs")
    for task in tasks:
        for i, id in enumerate(hadm_ids):
            print(f"Create batch for AWS")
            print(f"{task} on document {id} - {i+1}/{len(hadm_ids)}")
            build_all_variations_for_task(id, task, model=model, cdatetime=start_time)
            print(f"Pipeline completed - {id}")
    print("All pipelines completed")
    endtime = time.time() - start_time
    print(f"Time taken: {endtime}")


def run_batch_job():
    boto3 = {}
    bedrock = boto3.client(service_name="bedrock")
    inputDataConfig = {
        "s3InputDataConfig": {
            "s3Uri": "s3://info-leak-experiments/inputs/claude-3-5-sonnet-20240620-v1:0.jsonl"
        }
    }
    outputDataConfig = {
        "s3OutputDataConfig": {"s3Uri": "s3://info-leak-experiments/outputs/"}
    }
    response = bedrock.create_model_invocation_job(
        roleArn="arn:aws:iam::123456789012:role/MyBatchInferenceRole",
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        jobName="my-batch-job",
        inputDataConfig=inputDataConfig,
        outputDataConfig=outputDataConfig,
    )

    jobArn = response.get("jobArn")
    print(jobArn)


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
        default="claude-3-5-sonnet-20240620",
        choices=["claude-3-5-sonnet-20240620"],
    )
    parser.add_argument(
        "-f",
        "--flag",
        help="Choose a batch mode",
        default=BATCH_FLAGS[0],
        choices=BATCH_FLAGS,
    )
    parser.add_argument("-j", "--job_id", help="Choose a job identifier to check upon")
    parser.add_argument("-fi", "--file_id", help="Choose an output file identifier")
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

    if flag == "create-batch":
        if task == "legal_court":
            print("Starting legal court inference")
            legal_data = open_legal_data()
            # remove 5 for ICL
            ids = list(legal_data.keys())
            ids = ids[:-5]
            create_batch(hadm_ids=ids, tasks=[task], model=args.model)
        elif task == "cnn":
            print("Starting CNN inference")
            news_data = open_cnn_data()
            # remove 5 for ICL
            ids = list(news_data.keys())
            ids = ids[:-5]
            create_batch(hadm_ids=ids, tasks=[task], model=args.model)
        else:
            target_admission_ids = extract_hadm_ids_from_dir(
                "llama-3-8b-Instruct-bnb-4bit", "brief_hospital_course", "variation_1"
            )
            create_batch(
                hadm_ids=target_admission_ids[0:200], tasks=[task], model=args.model
            )
    elif flag == "retrieve":
        file_id = args.file_id
        with open(f"{BATCH_RESULTS_DIR}/{file_id}", "r") as file:
            lines = file.readlines()
            for line in lines:
                line_json = json.loads(line)
                id = line_json["recordId"]
                print(f"Document {id}")
                cid_data = id.split("_")
                file_id = cid_data.pop()
                variation = f"{cid_data.pop(0)}_{cid_data.pop(0)}"
                task = "_".join(cid_data)
                target_out_file = f"{RESULTS_DIR}/{model}/{variation}/{task}/{file_id}-discharge-inputs.txt"
                if not os.path.exists(f"{RESULTS_DIR}/{model}/{variation}/{task}"):
                    os.makedirs(f"{RESULTS_DIR}/{model}/{variation}/{task}")
                print(f"Output location {target_out_file}")
                with open(target_out_file, "w") as f:
                    f.write(
                        line_json["modelOutput"]["content"][0]["text"]
                    )