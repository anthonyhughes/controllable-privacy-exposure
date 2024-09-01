from constants import (
    BASELINE_SUMMARY_TASK,
    EVAL_MODELS,
    IN_CONTEXT_SUMMARY_TASK,
    METRICS,
    SUMMARY_TYPES,
)
from evaluate import load

from utils.dataset_utils import (
    extract_hadm_ids_from_dir,
    open_generated_summary,
    open_target_summary,
    reference_file_is_present,
)
from utils.pii_eval import run_privacy_eval, store_results
from timeit import default_timer as timer

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


def calculate_averages(results):
    """
    Calculate the average scores for the evaluation results
    """
    baseline_tasks = [f"{task}{BASELINE_SUMMARY_TASK}" for task in SUMMARY_TYPES]
    icl_tasks = [f"{task}{IN_CONTEXT_SUMMARY_TASK}" for task in SUMMARY_TYPES]
    avg_scores = {
        SUMMARY_TYPES[0]: {},
        SUMMARY_TYPES[1]: {},
        baseline_tasks[0]: {},
        baseline_tasks[1]: {},
        icl_tasks[0]: {},
        icl_tasks[1]: {},
    }
    for task in avg_scores.keys():
        print(f"Calculating average scores for task: {task}")
        for metric in METRICS:
            avg_scores[task][metric] = 0
            doc_count = 0
            for id in results.keys():
                if task in results[id] and metric in results[id][task]:
                    avg_scores[task][metric] += results[id][task][metric]
                    doc_count += 1
            avg_scores[task][metric] /= doc_count
    return avg_scores


def update_results_for_task(results, task, hadm_id, target_model):
    """
    Update the results for a given task
    """
    if reference_file_is_present(task, hadm_id) is True:
        eval_res = run_eval_for_document(task, hadm_id, target_model)
        if hadm_id not in results:
            results[hadm_id] = {}

        if task not in results[hadm_id]:
            results[hadm_id][task] = {}

        results[hadm_id][task] = eval_res
    else:
        print(f"Reference file not present for {task} - document: {hadm_id}")
    return results


def run_utility_eval(target_model):
    """
    Get the scores for utility
    """
    print("Running the utility evaluation pipeline")
    results = {}
    for task in SUMMARY_TYPES:
        baseline_task = f"{task}{BASELINE_SUMMARY_TASK}"
        print(f"Running evaluation for task: {baseline_task} and model: {target_model}")
        hadm_ids = extract_hadm_ids_from_dir(target_model, baseline_task)
        for i, hadm_id in enumerate(hadm_ids):
            print(
                f"Running evaluation for {baseline_task}, {target_model} and document: {hadm_id} - {i+1}/{len(hadm_ids)}"
            )
            print(f"Running evaluation for document: {hadm_id}")
            update_results_for_task(results, baseline_task, hadm_id, target_model)

        print(f"Running evaluation for task: {task} and model: {target_model}")
        hadm_ids = extract_hadm_ids_from_dir(target_model, task)
        for i, hadm_id in enumerate(hadm_ids):
            print(
                f"Running evaluation for {task}, {target_model} and document: {hadm_id} - {i+1}/{len(hadm_ids)}"
            )
            print(f"Running evaluation for document: {hadm_id}")
            update_results_for_task(results, task, hadm_id, target_model)

        icl_task = f"{task}{IN_CONTEXT_SUMMARY_TASK}"
        print(f"Running evaluation for task: {icl_task} and model: {target_model}")
        hadm_ids = extract_hadm_ids_from_dir(target_model, icl_task)
        for i, hadm_id in enumerate(hadm_ids):
            print(
                f"Running evaluation for {icl_task}, {target_model} and document: {hadm_id} - {i+1}/{len(hadm_ids)}"
            )
            print(f"Running evaluation for document: {hadm_id}")
            update_results_for_task(results, icl_task, hadm_id, target_model)

    store_results(results, target_model, "raw_utility")
    avg_results = calculate_averages(results)
    store_results(avg_results, target_model, "utility")


if __name__ == "__main__":
    start = timer()
    for target_model in EVAL_MODELS[0:1]:
        print(f"Running evaluation pipeline for model: {target_model}")
        run_utility_eval(target_model=target_model)
        run_privacy_eval(target_model=target_model)
    end = timer() - start
    print(f"Time to complete in secs: {end}")
