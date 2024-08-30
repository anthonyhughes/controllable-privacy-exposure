from constants import (
    BASELINE_SUMMARY_TASK,
    EVAL_MODELS,
    IN_CONTEXT_SUMMARY_TASK,
    MODELS,
    SUMMARY_TYPES,
)
from evaluate import load

from mimic.mimic_data import load_original_discharge_summaries
from utils.dataset_utils import (
    extract_hadm_ids,
    open_generated_summary,
    open_target_summary,
    result_file_is_present,
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


def calculate_averages(results, hadm_ids):
    """
    Calculate the average scores for the evaluation results
    """
    baseline_tasks = [f"{task}_baseline" for task in SUMMARY_TYPES]
    icl_tasks = [f"{task}_in_context_summary_task" for task in SUMMARY_TYPES]
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
        if task in results[hadm_ids[0]]:
            for metric in results[hadm_ids[0]][task]:
                avg_scores[task][metric] = 0
                for hadm_id in hadm_ids:
                    if hadm_id in results:
                        if task in results[hadm_id]:
                            avg_scores[task][metric] += results[hadm_id][task][metric]
                avg_scores[task][metric] /= len(hadm_ids)
    return avg_scores


def run_utility_eval(hadm_ids, target_model):
    """
    Get the scores for utility
    """
    print("Running the utility evaluation pipeline")
    results = {}
    for task in SUMMARY_TYPES:
        baseline_task = f"{task}{BASELINE_SUMMARY_TASK}"
        icl_task = f"{task}{IN_CONTEXT_SUMMARY_TASK}"
        print(f"Running evaluation for task: {task}")
        for i, hadm_id in enumerate(hadm_ids):
            print(f"Running evaluation for document: {hadm_id}")
            print(f"Document {i+1}/{len(hadm_ids)}")
            if result_file_is_present(task, hadm_id, target_model):
                if hadm_id not in results:
                    # eval_res = run_eval_for_document(task, hadm_id, target_model)
                    # baseline_eval_res = run_eval_for_document(
                    #     baseline_task, hadm_id, target_model
                    # )
                    icl_eval_res = run_eval_for_document(
                        icl_task, hadm_id, target_model
                    )
                    results[hadm_id] = {
                        task: {},
                        baseline_task: {},
                        icl_task: icl_eval_res,
                    }
                else:
                    results[hadm_id].update(
                        {
                            task: {},
                            baseline_task: {},
                            icl_task: icl_eval_res,
                        }
                    )
            else:
                print(f"Result file not present for document: {hadm_id}")
    store_results(results, target_model, "raw_utility")
    avg_results = calculate_averages(results, hadm_ids)
    store_results(avg_results, target_model, "utility")


if __name__ == "__main__":
    start = timer()
    original_discharge_summaries = load_original_discharge_summaries()
    target_admission_ids = extract_hadm_ids(
        original_discharge_summaries=original_discharge_summaries, n=10000
    )
    # remove the last 5 admission ids (preserved for ICL)
    target_admission_ids = target_admission_ids[:-5]
    for target_model in EVAL_MODELS[0:1]:
        print(f"Running evaluation pipeline for model: {target_model}")
        run_utility_eval(hadm_ids=target_admission_ids[0:100], target_model=target_model)
        run_privacy_eval(hadm_ids=target_admission_ids[0:100], target_model=target_model)
    end = timer() - start
    print(f"Time to complete in secs: {end}")
