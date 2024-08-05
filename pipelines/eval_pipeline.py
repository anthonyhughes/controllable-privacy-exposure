import json
from constants import (
    OPEN_AI_RESULTS_DIR,
    PSEUDO_TARGETS_ROOT,
    SUMMARY_TYPES,
    EXAMPLE_ADMISSION_IDS,
)
from evaluate import load

from mimic.mimic_data import load_original_discharge_summaries
from utils.dataset_utils import extract_hadm_ids

bertscore = load("bertscore")
rouge_eval = load("rouge")


def open_generated_summary(task, hadm_id):
    """
    Load the generated summary for a document
    """
    with open(f"{OPEN_AI_RESULTS_DIR}/{task}/{hadm_id}_{task}_summary.txt", "r") as f:
        return f.read()


def open_target_summary(task, hadm_id):
    """
    Load the target summary for a document
    """
    with open(f"{PSEUDO_TARGETS_ROOT}{task}/{hadm_id}-target.txt", "r") as f:
        return f.read()


def run_eval_for_document(task, hadm_id):
    """
    Run evaluation for a single document
    """
    generated_summary = open_generated_summary(task, hadm_id)
    ground_truth_summary = open_target_summary(task, hadm_id)
    result = rouge_eval.compute(
        predictions=[generated_summary], references=[ground_truth_summary]
    )
    bertscore_result = bertscore.compute(
        predictions=[generated_summary], references=[ground_truth_summary], lang="en"
    )
    result["bertscore"] = bertscore_result["f1"][0]
    return result


def calculate_averages(results, hadm_ids):
    """
    Calculate the average scores for the evaluation results
    """
    avg_scores = {
        SUMMARY_TYPES[0]: {},
        SUMMARY_TYPES[1]: {},
    }
    for task in SUMMARY_TYPES:
        print(f"Calculating average scores for task: {task}")
        for metric in results[hadm_ids[0]][task]:
            avg_scores[task][metric] = 0
            for hadm_id in hadm_ids:
                avg_scores[task][metric] += results[hadm_id][task][metric]
            avg_scores[task][metric] /= len(hadm_ids)
    with open(f"data/results.json", "w") as f:
        json.dump(avg_scores, f, indent=4)


def run(hadm_ids):
    print("Running evaluation pipeline")
    results = {}
    for task in SUMMARY_TYPES:
        print(f"Running evaluation for task: {task}")
        for hadm_id in hadm_ids:
            if hadm_id not in results:
                results[hadm_id] = {task: run_eval_for_document(task, hadm_id)}
            else:
                results[hadm_id].update({task: run_eval_for_document(task, hadm_id)})
    # print(results)
    calculate_averages(results, hadm_ids)


if __name__ == "__main__":
    original_discharge_summaries = load_original_discharge_summaries()
    target_admission_ids = extract_hadm_ids(
        original_discharge_summaries=original_discharge_summaries, n=100
    )
    run(hadm_ids=target_admission_ids)
