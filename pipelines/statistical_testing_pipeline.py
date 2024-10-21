import argparse
import json
import os
import numpy as np
from scipy import stats

from constants import (
    BASELINE_SUMMARY_TASK,
    BRIEF_HOSPITAL_COURSE,
    EVAL_MODELS,
    IN_CONTEXT_SUMMARY_TASK,
    PRIV_SUMMARY_TASK,
    SANI_SUMM_SUMMARY_TASK,
    SANITIZE_TASK,
    SUMMARY_TYPES,
    UTILITY_RESULTS_DIR,
)


def get_task_data(
    target_model,
    task=BRIEF_HOSPITAL_COURSE,
    comparator_task=f"{PRIV_SUMMARY_TASK}",
    metric="rogue_1",
):
    target_task = f"{task}{BASELINE_SUMMARY_TASK}"
    comparator_task = f"{task}{comparator_task}"
    variations = ["variation_1", "variation_2", "variation_3"]
    task_averages = []
    comparator_averages = []
    for variation_name in variations:
        target_file = (
            f"{UTILITY_RESULTS_DIR}/{variation_name}_{task}_utility/{target_model}.json"
        )
        if os.path.exists(target_file):
            with open(target_file, "r") as f:
                data = json.load(f)
                bhc_data = data[target_task]
                task_averages.append(bhc_data[metric])
                comparator_data = data[comparator_task]
                comparator_averages.append(comparator_data[metric])
        else:
            print(f"Missing file {target_file}")
    return task_averages, comparator_averages


def t_test_calc(data: list, comparator_data: list):
    # Calculate the differences
    data = np.array(data)
    comparator_data = np.array(comparator_data)
    differences = comparator_data - data

    # Check for normality (Shapiro-Wilk test on differences)
    stat, p_value_normality = stats.shapiro(differences)

    # Perform appropriate test based on normality
    if p_value_normality > 0.05:
        # Perform paired t-test if normality is not rejected
        t_stat, p_value_ttest = stats.ttest_rel(data, comparator_data)
        test_used = "Paired t-test"
        result = p_value_ttest
    else:
        # Perform Wilcoxon signed-rank test if normality is rejected
        w_stat, p_value_wilcoxon = stats.wilcoxon(data, comparator_data)
        test_used = "Wilcoxon signed-rank test"
        result = p_value_wilcoxon

    # Calculate mean difference
    mean_difference = np.mean(differences)

    # Determine the direction of the change
    if mean_difference > 0:
        direction = "positive (increase)"
    elif mean_difference < 0:
        direction = "negative (decrease)"
    else:
        direction = "no overall change"

    # Output the results
    print(f"Test used: {test_used}")
    print(f"P-value: {result}")
    return result, mean_difference, direction


def run():
    # Scores from the first table
    for model in EVAL_MODELS[0:2]:
        all_tasks_results = []
        for task in SUMMARY_TYPES:
            task_result = []
            for sub_task in [PRIV_SUMMARY_TASK, IN_CONTEXT_SUMMARY_TASK, SANI_SUMM_SUMMARY_TASK]:
                sub_task_result = []
                print(f"{sub_task}")
                if sub_task == SANITIZE_TASK:
                    continue
                for metric in ["rouge1", "rouge2", "rougeL", "bertscore"]:
                    print(
                        f"Comparing {task}_baseline with {task}{sub_task} for metric {metric}"
                    )
                    task_averages, comparator_averages = get_task_data(
                        target_model=model,
                        task=task,
                        comparator_task=sub_task,
                        metric=metric,
                    )
                    # print(task_averages, comparator_averages)
                    result, _, direction = t_test_calc(
                        task_averages, comparator_averages
                    )
                    signifiance_found = "true" if (result < 0.05) else "false"
                    print(f"Signifiance found {signifiance_found} in {direction}\n")
                    sub_task_result.append(
                        (task, sub_task, metric, signifiance_found, direction)
                    )
                task_result.append(sub_task_result)
            all_tasks_results.append(task_result)
        with open(f"data/stats_results/{model}.json", "w") as f:
            json.dump(all_tasks_results, f, indent=4)


if __name__ == "__main__":
    print("Starting statistical testing")
    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        help="Choose a target model for inference",
        default=EVAL_MODELS,
        choices=EVAL_MODELS,
    )
    args = parser.parse_args()
    run()
