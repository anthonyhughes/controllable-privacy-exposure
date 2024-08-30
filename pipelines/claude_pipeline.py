import os
from anthropic import Anthropic
import time
from constants import (
    RESULTS_DIR,
    SUMMARY_TYPES,
    TASK_SUFFIXES,
)
from mimic.mimic_data import load_original_discharge_summaries, get_ehr_and_summary
from utils.dataset_utils import extract_hadm_ids
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


def run(
    hadm_ids,
    model="claude-3-5-sonnet-20240620",
    tasks_suffixes=None,
    icl_hadm_ids=None,
):
    print("Running the claude pipeline")

    client = Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )

    for task in SUMMARY_TYPES:
        for i, id in enumerate(hadm_ids):
            print(
                f"Running pipeline for {task} on patient {id} - {i+1}/{len(hadm_ids)}"
            )
            if id in ["28664981", "21441082"]:
                continue
            all_inference_tasks(
                id,
                task,
                prompt_prefix_for_task,
                inference_fnc=inference,
                client=client,
                tasks_suffixes=tasks_suffixes,
                model=model,
                icl_hadm_ids=icl_hadm_ids,
            )            
            print(f"Pipeline completed - {id}")
    print("All pipelines completed")


if __name__ == "__main__":
    start_time = time.time()
    original_discharge_summaries = load_original_discharge_summaries()
    target_admission_ids = extract_hadm_ids(
        original_discharge_summaries=original_discharge_summaries, n=10000
    )
    icl_hadm_ids = target_admission_ids[-1:]
    target_admission_ids = target_admission_ids[:-5]
    run(
        hadm_ids=target_admission_ids[0:100],
        model="claude-3-5-sonnet-20240620",
        tasks_suffixes=TASK_SUFFIXES,
        icl_hadm_ids=icl_hadm_ids
    )
    endtime = time.time() - start_time
    print(f"Time taken: {endtime}")
