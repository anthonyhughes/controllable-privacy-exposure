import os
import time
from openai import OpenAI
from constants import (
    RE_ID_EXAMPLES_ROOT,
    PSEUDO_TARGETS_ROOT,
    RESULTS_DIR,
    SUMMARY_TYPES,
    EXAMPLE_ADMISSION_IDS,
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
        ehr = f.readlines()
    with open(
        f"{PSEUDO_TARGETS_ROOT}{task}/{hadm_id}-target.txt",
        "r",
    ) as f:
        summary = f.readlines()
    return ehr, summary


def inference(client, task, hadm_id):
    """Run the openai query"""
    ehr, summary = get_ehr_and_summary(task, hadm_id)
    prompt = {
        "role": "user",
        "content": f"""
                {prompt_prefix_for_task[task]}
                {ehr}
                """,
    }
    chat_completion = client.chat.completions.create(
        messages=[prompt],
        model="gpt-4o-mini",
    )
    summary = chat_completion.choices[0].message.content
    return summary


def save_result(openai_result, task, hadm_id):
    """Save result"""
    if not os.path.exists(f"data/results/openai/{task}"):
        os.makedirs(f"data/results/openai/{task}")
    with open(
        f"{RESULTS_DIR}/openai/{task}/{hadm_id}_{task}_summary.txt",
        "w",
    ) as f:
        f.write(openai_result)


def run(hadm_ids):
    start_time = time.time()
    print("Running the pipelines")
    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    for task in SUMMARY_TYPES:
        for id in hadm_ids:
            print(f"Running pipeline for {task} on patient {id}")
            if not os.path.exists(f"{RESULTS_DIR}/openai/{task}/{id}_{task}_summary.txt"):
                print("Starting inference")
                result = inference(client, task, hadm_id=id)
                save_result(result, task, hadm_id=id)
                print("Pipeline completed")
    print("All pipelines completed")
    endtime = time.time() - start_time
    print(f"Time taken: {endtime}")


if __name__ == "__main__":
    original_discharge_summaries = load_original_discharge_summaries()
    target_admission_ids = extract_hadm_ids(
        original_discharge_summaries=original_discharge_summaries, n=100
    )
    run(hadm_ids=target_admission_ids)
