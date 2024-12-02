from datetime import datetime
import json
import os
from constants import (
    BASELINE_SUMMARY_TASK,
    IN_CONTEXT_SUMMARY_TASK,
    METRICS,
    SANI_SUMM_SUMMARY_TASK,
    SUMM_SANN_SUMMARY_TASK,
    TASK_SUFFIXES,
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

from utils.prompt_variations import variations

bertscore = load("bertscore")
rouge_eval = load("rouge")


def print_results_to_latex(target_model, task):
    """
    Print to latex for easy copy&paste
    """
    with open(f"{UTILITY_RESULTS_DIR}/{task}_utility/{target_model}.json") as f:
        json_data = json.load(f)

    for key, metrics in json_data.items():
        metrics_str = " & ".join(f"{metrics[metric]:.2f}" for metric in metrics)

        with open(f"{UTILITY_RESULTS_DIR}/latex/utility.txt", "a") as f:
            f.write(f"{datetime.now()} \\\\ \n")
            f.write(f"{key}-{target_model} & {metrics_str} \\\\ \n")


def calculate_average_per_variation(results, task, variation):
    """
    Calculate the average scores for the evaluation results
    """
    avg_scores = {}

    baseline_task = f"{task}{BASELINE_SUMMARY_TASK}"
    icl_task = f"{task}{IN_CONTEXT_SUMMARY_TASK}"
    sani_summ_task = f"{task}{SANI_SUMM_SUMMARY_TASK}"
    summ_sann_task = f"{task}{SUMM_SANN_SUMMARY_TASK}"

    tasks = [task, baseline_task, icl_task, sani_summ_task, summ_sann_task]

    # Add these tasks to the avg_scores dictionary
    avg_scores[baseline_task] = {}
    avg_scores[icl_task] = {}
    avg_scores[task] = {}
    avg_scores[sani_summ_task] = {}
    avg_scores[summ_sann_task] = {}

    for task in tasks:
        for metric in METRICS:
            print(
                f"Calculating average scores for each task {task} per metric {metric}"
            )
            avg_scores[task][metric] = 0
            doc_count = 0
            if variation in results:
                results_variant = results[variation]
                for id in results_variant.keys():
                    if task in results_variant[id] and metric in results_variant[id][task]:
                        avg_scores[task][metric] += results_variant[id][task][metric]
                        doc_count += 1
                if doc_count > 0:
                    avg_scores[task][metric] /= doc_count
    return avg_scores


def calculate_overall_averages(results):    
    result = {}

    # Iterate over each dictionary in the list
    for d in results:
        # Iterate over each key (e.g., 'legal_court_baseline')
        for category, metrics in d.items():
            if category not in result:
                result[category] = {metric: 0.0 for metric in metrics}
            
            # Sum up the values for each metric
            for metric, value in metrics.items():
                result[category][metric] += value

    # Calculate the average for each metric
    num_dicts = len(results)
    for category, metrics in result.items():
        for metric in metrics:
            result[category][metric] /= num_dicts

    return result


def run_eval_for_document(task, hadm_id, target_model, variation):
    """
    Run evaluation for a single document
    """
    generated_summary = open_generated_summary(task, hadm_id, target_model, variation)
    ground_truth_summary = open_target_summary(task, hadm_id)
    result = rouge_eval.compute(
        predictions=[generated_summary], references=[ground_truth_summary]
    )
    bertscore_result = bertscore.compute(
        predictions=[generated_summary], references=[ground_truth_summary], lang="en"
    )
    result["bertscore"] = bertscore_result["f1"][0]
    return result


def update_results_for_task(results, task, hadm_id, target_model, variation):
    """
    Update the results for a given task
    """
    if reference_file_is_present(task, hadm_id) is True:
        eval_res = run_eval_for_document(task, hadm_id, target_model, variation)
        if variation not in results:
            results[variation] = {}

        if hadm_id not in results[variation]:
            results[variation][hadm_id] = {}

        if task not in results[variation][hadm_id]:
            results[variation][hadm_id][task] = {}

        results[variation][hadm_id][task] = eval_res
    else:
        print(f"Reference file not present for {task} - document: {hadm_id}")
    return results


def run_utility_eval(target_model, tasks=SUMMARY_TYPES, sub_tasks=TASK_SUFFIXES):
    """
    Get the scores for utility
    """
    print("Running the utility evaluation pipeline")    
    for task in tasks:        
        results = {}
        variant_results = []
        for variation_prompt in variations:                
            variation = variation_prompt["name"]
            baseline_task = f"{task}{BASELINE_SUMMARY_TASK}"
            print(f"Running evaluation for task: {baseline_task} and model: {target_model}")
            hadm_ids = extract_hadm_ids_from_dir(target_model, baseline_task, variation)
            if BASELINE_SUMMARY_TASK in sub_tasks:
                for i, hadm_id in enumerate(hadm_ids):
                    print(f"Completed: {i+1}/{len(hadm_ids)}")
                    print(
                        f"Running evaluation for {baseline_task}, model: {target_model}, document: {hadm_id}, variation: {variation}"
                    )
                    print(f"Running evaluation for document: {hadm_id}")
                    update_results_for_task(results, baseline_task, hadm_id, target_model, variation)
            
            if "" in sub_tasks:
                print(f"Running evaluation for task: {task} and model: {target_model}")
                hadm_ids = extract_hadm_ids_from_dir(target_model, task, variation)
                for i, hadm_id in enumerate(hadm_ids):
                    print(f"Completed: {i+1}/{len(hadm_ids)}")
                    print(
                        f"Running evaluation for {task}, {target_model}, document: {hadm_id}, variation: {variation}"
                    )
                    print(f"Running evaluation for document: {hadm_id}")
                    update_results_for_task(results, task, hadm_id, target_model, variation)
            
            if IN_CONTEXT_SUMMARY_TASK in sub_tasks:
                icl_task = f"{task}{IN_CONTEXT_SUMMARY_TASK}"
                print(f"Running evaluation for task: {icl_task} and model: {target_model}")
                hadm_ids = extract_hadm_ids_from_dir(target_model, icl_task, variation)
                for i, hadm_id in enumerate(hadm_ids):
                    print(f"Completed: {i+1}/{len(hadm_ids)}")
                    print(
                        f"Running evaluation for {icl_task}, {target_model}, document: {hadm_id}, variation: {variation}"
                    )
                    print(f"Running evaluation for document: {hadm_id}")
                    update_results_for_task(results, icl_task, hadm_id, target_model, variation)

            if SANI_SUMM_SUMMARY_TASK in sub_tasks:
                sani_summ_task = f"{task}{SANI_SUMM_SUMMARY_TASK}"
                print(f"Running evaluation for task: {sani_summ_task} and model: {target_model}")
                hadm_ids = extract_hadm_ids_from_dir(target_model, sani_summ_task, variation)
                for i, hadm_id in enumerate(hadm_ids):
                    print(f"Completed: {i+1}/{len(hadm_ids)}")
                    print(
                        f"Running evaluation for {sani_summ_task}, model: {target_model}, document: {hadm_id}, variation: {variation}"
                    )
                    print(f"Running evaluation for document: {hadm_id}")
                    update_results_for_task(results, sani_summ_task, hadm_id, target_model, variation)

            if SUMM_SANN_SUMMARY_TASK in sub_tasks:
                summ_sann_task = f"{task}{SUMM_SANN_SUMMARY_TASK}"
                print(f"Running evaluation for task: {summ_sann_task} and model: {target_model}")
                hadm_ids = extract_hadm_ids_from_dir(target_model, summ_sann_task, variation)
                for i, hadm_id in enumerate(hadm_ids[0:10]):
                    print(f"Completed: {i+1}/{len(hadm_ids)}")
                    print(
                        f"Running evaluation for {summ_sann_task}, model: {target_model}, document: {hadm_id}, variation: {variation}"
                    )
                    print(f"Running evaluation for document: {hadm_id}")
                    update_results_for_task(results, summ_sann_task, hadm_id, target_model, variation)
            
            # Store the averages per variant
            avg_results_for_variant = calculate_average_per_variation(results, task=task, variation=variation)        
            variant_results.append(avg_results_for_variant)
            store_utility_results(avg_results_for_variant, target_model, f"{variation}_{task}_utility")
        # store all variations and all document utility results
        overall_averages = calculate_overall_averages(variant_results)
        store_utility_results(overall_averages, target_model, f"{task}_final_utility")
