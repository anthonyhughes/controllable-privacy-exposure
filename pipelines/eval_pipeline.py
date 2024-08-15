from constants import (
    MODELS,
    SUMMARY_TYPES,
)
from evaluate import load

from mimic.mimic_data import load_original_discharge_summaries
from utils.dataset_utils import (
    extract_hadm_ids,
    open_generated_summary,
    open_target_summary,
)
from utils.pii_eval import run_privacy_eval, store_results

bertscore = load("bertscore")
rouge_eval = load("rouge")


def run_eval_for_document(task, hadm_id, target_model):
    """
    Run evaluation for a single document
    """
    generated_summary = open_generated_summary(task, hadm_id, target_model)
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
    return avg_scores


def run_utility_eval(hadm_ids, target_model):
    """
    Get the scores for utility
    """
    print("Running evaluation pipeline")
    results = {}
    for task in SUMMARY_TYPES:
        print(f"Running evaluation for task: {task}")
        for hadm_id in hadm_ids:
            if hadm_id not in results:
                results[hadm_id] = {
                    task: run_eval_for_document(task, hadm_id, target_model)
                }
            else:
                results[hadm_id].update(
                    {task: run_eval_for_document(task, hadm_id, target_model)}
                )
    results = calculate_averages(results, hadm_ids)
    store_results(results, target_model, "utility")


if __name__ == "__main__":
    original_discharge_summaries = load_original_discharge_summaries()
    target_admission_ids = extract_hadm_ids(
        original_discharge_summaries=original_discharge_summaries, n=100
    )
    for target_model in MODELS:
        print(f"Running evaluation pipeline for model: {target_model}")
        run_utility_eval(hadm_ids=target_admission_ids, target_model=target_model)
        run_privacy_eval(hadm_ids=target_admission_ids, target_model=target_model)
