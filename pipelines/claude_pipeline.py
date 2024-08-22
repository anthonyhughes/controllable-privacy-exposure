import os
from anthropic import Anthropic
import time
from constants import (
    RE_ID_EXAMPLES_ROOT,
    PSEUDO_TARGETS_ROOT,
    RESULTS_DIR,
    SUMMARY_TYPES,
)
from mimic.mimic_data import load_original_discharge_summaries
from utils.dataset_utils import extract_hadm_ids
from utils.prompts import prompt_prefix_for_task


def get_ehr_and_summary(task, hadm_id):
    """Get the EHR and summary"""
    with open(
        f"{RE_ID_EXAMPLES_ROOT}{task}/{hadm_id}-discharge-inputs.txt",
        "r",
    ) as f:
        ehr = f.read()
    with open(
        f"{PSEUDO_TARGETS_ROOT}{task}/{hadm_id}-target.txt",
        "r",
    ) as f:
        summary = f.read()
    return ehr, summary


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


def save_result(result, task, hadm_id, model):
    """Save result"""
    if not os.path.exists(f"data/results/{model}/{task}"):
        os.makedirs(f"data/results/{model}/{task}")
    with open(
        f"{RESULTS_DIR}/{model}/{task}/{hadm_id}-discharge-inputs.txt",
        "w",
    ) as f:
        f.write(result)


def run(hadm_ids, model):
    print("Running the claude pipeline")

    client = Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )

    for task in SUMMARY_TYPES:
        for i, id in enumerate(hadm_ids):
            print(f"Running pipeline for {task} on patient {id} - {i+1}/{len(hadm_ids)}")
            if id in ['28664981']: continue
            if not os.path.exists(
                f"{RESULTS_DIR}/{model}/{task}/{id}-discharge-inputs.txt"
            ):
                print(f"Starting inference - {id}")
                baseline_prompt = prompt_prefix_for_task[
                    f"{task}_baseline_summary_task"
                ]
                baseline_result = inference(
                    client, task, hadm_id=id, model=model, prompt=baseline_prompt
                )
                main_prompt = prompt_prefix_for_task[task]
                pseudonymised_result = inference(
                    client, task, hadm_id=id, model=model, prompt=main_prompt
                )
                save_result(pseudonymised_result, task, hadm_id=id, model=model)
                save_result(
                    baseline_result, f"{task}_baseline", hadm_id=id, model=model
                )
                print(f"Pipeline completed - {id}")
                # sleep for 10 seconds to avoid rate limiting
                time.sleep(10)
            else:
                print(f"Skipping inference as result already exists - {id}")
    print("All pipelines completed")


if __name__ == "__main__":
    start_time = time.time()
    original_discharge_summaries = load_original_discharge_summaries()
    target_admission_ids = extract_hadm_ids(
        original_discharge_summaries=original_discharge_summaries, n=100
    )
    run(hadm_ids=target_admission_ids, model="claude-3-5-sonnet-20240620")
    endtime = time.time() - start_time
    print(f"Time taken: {endtime}")
