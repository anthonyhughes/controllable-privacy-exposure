from datetime import datetime
import json
import os
from constants import (
    BASELINE_SUMMARY_TASK,
    IN_CONTEXT_SUMMARY_TASK,
    METRICS,
    UTILITY_RESULTS_DIR,
    SUMMARY_TYPES,
)
from evaluate import load

from utils.dataset_utils import (
    extract_hadm_ids_from_dir,
    open_generated_summary,
    open_target_summary,
    reference_file_is_present,
    store_utility_results,
)

bertscore = load("bertscore")
rouge_eval = load("rouge")


def print_results_to_latex(target_model, task):
    with open(f"{UTILITY_RESULTS_DIR}/{task}_utility/{target_model}.json") as f:
        json_data = json.load(f)

    for key, metrics in json_data.items():
        metrics_str = " & ".join(f"{metrics[metric]:.2f}" for metric in metrics)

        with open(f"{UTILITY_RESULTS_DIR}/latex/utility.txt", "a") as f:
            f.write(f"{datetime.now()} \\\\ \n")
            f.write(f"{key}-{target_model} & {metrics_str} \\\\ \n")


def calculate_averages(results, task):
    """
    Calculate the average scores for the evaluation results
    """
    avg_scores = {}

    baseline_task = f"{task}{BASELINE_SUMMARY_TASK}"
    icl_task = f"{task}{IN_CONTEXT_SUMMARY_TASK}"

    tasks = [task, baseline_task, icl_task]

    # Add these tasks to the avg_scores dictionary
    avg_scores[baseline_task] = {}
    avg_scores[icl_task] = {}
    avg_scores[task] = {}

    for task in tasks:
        for metric in METRICS:
            print(
                f"Calculating average scores for each task {task} per metric {metric}"
            )
            avg_scores[task][metric] = 0
            doc_count = 0
            for id in results.keys():
                if task in results[id] and metric in results[id][task]:
                    avg_scores[task][metric] += results[id][task][metric]
                    doc_count += 1
            if doc_count > 0:
                avg_scores[task][metric] /= doc_count
    return avg_scores


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


def run_utility_eval(target_model, tasks=SUMMARY_TYPES):
    """
    Get the scores for utility
    """
    print("Running the utility evaluation pipeline")
    results = {}
    for task in tasks:
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

        store_utility_results(results, target_model, f"{task}_raw_utility")
        avg_results = calculate_averages(results, task=task)
        store_utility_results(avg_results, target_model, f"{task}_utility")
        # print_results_to_latex(target_model, task=task)
