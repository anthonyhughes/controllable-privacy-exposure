from constants import (
    OPEN_AI_RESULTS_DIR,
    PSEUDO_TARGETS_ROOT,
    SUMMARY_TYPES,
    EXAMPLE_ADMISSION_IDS,
)
from evaluate import load


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
    result = open_generated_summary(task, hadm_id)
    target = open_target_summary(task, hadm_id)
    rouge_eval = load("rouge")
    rouge_result = rouge_eval.compute(predictions=[result], references=[target])
    return rouge_result


def calculating_averages(results):
    """
    Calculate the average scores for the evaluation results
    """
    for task in SUMMARY_TYPES:
        print(f"Calculating average scores for task: {task}")
        avg_scores = {}
        for metric in results[EXAMPLE_ADMISSION_IDS[0]][task]:
            avg_scores[metric] = 0
            for hadm_id in EXAMPLE_ADMISSION_IDS:
                avg_scores[metric] += results[hadm_id][task][metric]
            avg_scores[metric] /= len(EXAMPLE_ADMISSION_IDS)
        print(avg_scores)


def run():
    print("Running evaluation pipeline")
    results = {}
    for task in SUMMARY_TYPES:
        print(f"Running evaluation for task: {task}")
        for hadm_id in EXAMPLE_ADMISSION_IDS:
            if hadm_id not in results:
                results[hadm_id] = {task: run_eval_for_document(task, hadm_id)}
            else:
                results[hadm_id].update({task: run_eval_for_document(task, hadm_id)})
    # print(results)    
    calculating_averages(results)


if __name__ == "__main__":
    run()
