import json
import os

import numpy as np
from constants import (
    EVAL_MODELS,
    FINAL_PRIVACY_RESULTS_DIR,
    FINAL_REID_RESULTS_DIR,
    PRIVACY_RESULTS_DIR,
    UTILITY_RESULTS_DIR,
)


reid_clinical_files = [
    "claude-3-5-sonnet-20240620-reidentification_results-20241007-093020.json",
    "gpt-4o-mini-reidentification_results-20241008-093250.json",
    "Meta-Llama-3.1-70B-Instruct-bnb-4bit-reidentification_results-20241008-092100.json",
    "llama-3-8b-Instruct-bnb-4bit-reidentification_results-20241008-095559.json",
    "mistral-7b-instruct-v0.3-bnb-4bit-reidentification_results-20241008-091200.json",
    "llamonymous-3-8b-bnb-4bit-reidentification_results-20241108-153742.json",
    "mistralymous-7b-bnb-4bit-reidentification_results-20241108-145140.json",
    "llamonymous-3-70b-bnb-4bit-reidentification_results-20241114-132820.json",
]

reid_non_clinical_files = [
    "claude-3-5-sonnet-20240620-reidentification_results-20241015-114557.json",
    "gpt-4o-mini-reidentification_results-20241015-115757.json",
    "Meta-Llama-3.1-70B-Instruct-bnb-4bit-reidentification_results-20241015-122234.json",
    "llama-3-8b-Instruct-bnb-4bit-reidentification_results-20241015-121513.json",
    "mistral-7b-instruct-v0.3-bnb-4bit-reidentification_results-20241015-120220.json",
    "llamonymous-3-8b-bnb-4bit-reidentification_results-20241108-153749.json",
    "mistralymous-7b-bnb-4bit-reidentification_results-20241108-145147.json",
    "llamonymous-3-70b-bnb-4bit-reidentification_results-20241114-132828.json",
]


def gen_data_for_all_properties_comparison():
    heat_datasets = []
    target_tasks = [
        "brief_hospital_course",
        "cnn",
        "discharge_instructions",
        "legal_court",
    ]
    for target_task in target_tasks:
        if target_task == "cnn" or target_task == "legal_court":
            files = reid_non_clinical_files
        else:
            files = reid_clinical_files

        heat_data = np.zeros((3, len(files)))
        for i, file in enumerate(files):
            with open(f"{FINAL_REID_RESULTS_DIR}/{file}", "r") as f:
                data = json.load(f)
                tasks = [key for key in data.keys() if target_task in key]
                for j, task in enumerate(tasks):
                    task_stats = data[task]
                    p_recall = task_stats["PERSON"]["recall"]
                    d_recall = task_stats["DATE"]["recall"]
                    o_recall = task_stats["ORG"]["recall"]
                    recall = (p_recall + d_recall + o_recall) / 3
                    heat_data[j][i] = recall
        heat_datasets.append(heat_data)
    return heat_datasets, target_tasks


def gen_data_for_property_comparison(property_name):
    heat_datasets = []
    target_tasks = [
        "brief_hospital_course",
        "cnn",
        "discharge_instructions",
        "legal_court",
    ]
    if property_name == "PERSON":
        target_tasks
        target_tasks = target_tasks[:3]
    for target_task in target_tasks:
        if target_task == "cnn" or target_task == "legal_court":
            files = reid_non_clinical_files
        else:
            files = reid_clinical_files

        heat_data = np.zeros((3, len(files)))
        for i, file in enumerate(files):
            with open(f"{FINAL_REID_RESULTS_DIR}/{file}", "r") as f:
                data = json.load(f)
                tasks = [key for key in data.keys() if target_task in key]
                for j, task in enumerate(tasks):
                    task_stats = data[task]
                    p_recall = task_stats[property_name]["recall"]
                    heat_data[j][i] = p_recall
        heat_datasets.append(heat_data)
    return heat_datasets, target_tasks


def handle_key(key):
    if "baseline" in key:
        return "Baseline"
    elif "_in_context" in key:
        return "1 Shot"
    elif "_sani_summ" in key:
        return "Sanitize and Summarize"
    else:
        return "0 Shot"


def find_privacy_file(model):
    privacy_files = os.listdir(FINAL_PRIVACY_RESULTS_DIR)
    for file in privacy_files:
        if model in file:
            return file
    return None


def gen_data_for_ptr_utility(utility_metric, privacy_metric="private_token_ratio"):
    tmp_data = {}
    for model in [
        "gpt-4o-mini",
        "claude-3-5-sonnet-20240620",
        "mistral-7b-instruct-v0.3-bnb-4bit",
        "mistralymous-7b-bnb-4bit",
        "Meta-Llama-3.1-70B-Instruct-bnb-4bit",
        "llamonymous-3-70b-bnb-4bit",
        "llama-3-8b-Instruct-bnb-4bit",
        "llamonymous-3-8b-bnb-4bit",
    ]:
        if model not in tmp_data:
            tmp_data[model] = {}
        for i, task in enumerate(
            [
                "brief_hospital_course",
                "cnn",
                "discharge_instructions",
                "legal_court",
            ]
        ):
            utility_file = f"{UTILITY_RESULTS_DIR}/{task}_final_utility/{model}.json"
            privacy_file = find_privacy_file(model)
            with open(utility_file, "r") as f:
                data = json.load(f)
                with open(f"{FINAL_PRIVACY_RESULTS_DIR}/{privacy_file}", "r") as f:
                    privacy_data = json.load(f)
                    for key in data.keys():
                        # if not "_in_context" or "_sani_" in key:
                        #     continue
                        if "baseline" in key:
                            continue
                        nkey = handle_key(key)
                        if nkey not in tmp_data[model]:
                            tmp_data[model][nkey] = []

                        tmp_data[model][nkey].append(
                            (
                                data[key][utility_metric],
                                privacy_data[key]["variation_1"][privacy_metric],
                            )
                        )

    return tmp_data
