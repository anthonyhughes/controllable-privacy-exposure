import json
import os
import random
from datasets import load_dataset

import pandas as pd

from constants import (
    BASELINE_SUMMARY_TASK,
    ICL_EXAMPLES_ROOT,
    IN_CONTEXT_SUMMARY_TASK,
    LEGAL_EXAMPLES_ROOT,
    RE_ID_EXAMPLES_ROOT,
    RESULTS_DIR,
    PSEUDO_TARGETS_ROOT,
    SANI_SUMM_SUMMARY_TASK,
    SUMMARY_TYPES,
    TRAIN_DISCHARGE_ME,
    RE_ID_TARGETS_ROOT,
    UTILITY_RESULTS_DIR,
    PRIVACY_RESULTS_DIR,
)


def create_missing_output_folders(target_model, v_name, summary_type):
    if os.path.exists(f"{RESULTS_DIR}/{target_model}/{v_name}/{summary_type}") is False:
        print("Creating a results folder")
        os.makedirs(
            f"{RESULTS_DIR}/{target_model}/{v_name}/{summary_type}", exist_ok=True
        )

    if (
        os.path.exists(
            f"{RESULTS_DIR}/{target_model}/{v_name}/{summary_type}{BASELINE_SUMMARY_TASK}"
        )
        is False
    ):
        print("Creating the results folder")
        os.makedirs(
            f"{RESULTS_DIR}/{target_model}/{v_name}/{summary_type}{BASELINE_SUMMARY_TASK}",
            exist_ok=True,
        )

    if (
        os.path.exists(
            f"{RESULTS_DIR}/{target_model}/{v_name}/{summary_type}{IN_CONTEXT_SUMMARY_TASK}"
        )
        is False
    ):
        print("Creating the results folder")
        os.makedirs(
            f"{RESULTS_DIR}/{target_model}/{v_name}/{summary_type}{IN_CONTEXT_SUMMARY_TASK}",
            exist_ok=True,
        )

    if (
        os.path.exists(
            f"{RESULTS_DIR}/{target_model}/{v_name}/{summary_type}{SANI_SUMM_SUMMARY_TASK}"
        )
        is False
    ):
        print("Creating the results folder")
        os.makedirs(
            f"{RESULTS_DIR}/{target_model}/{v_name}/{summary_type}{SANI_SUMM_SUMMARY_TASK}",
            exist_ok=True,
        )
        os.makedirs(
            f"{RESULTS_DIR}/{target_model}/{v_name}/{summary_type}_sanitized",
            exist_ok=True,
        )


def load_cnn_dataset():
    """"""
    dataset = load_dataset("ccdv/cnn_dailymail", "3.0.0", trust_remote_code=True)
    return dataset


def update_article_prefix(dataset, prefix):
    """Update the article prefix"""
    dataset = dataset.map(lambda example: {"article": prefix + example["article"]})
    return dataset


def extract_hadm_ids(original_discharge_summaries, n=100):
    """Extract the first n admission ids as a list"""
    if n > 0:
        hadm_df = original_discharge_summaries.head(n)
    else:
        hadm_df = original_discharge_summaries
    return list(hadm_df["hadm_id"])


def extract_hadm_ids_from_dir(model, task, variation):
    """Extract the hadm ids from the results directory"""
    files = os.listdir(f"{RESULTS_DIR}/{model}/{variation}/{task}")
    hadm_ids = []
    for file in files:
        if file.startswith("."):
            continue
        else:
            hadm_ids.append(file.split("-")[0])
    return hadm_ids


def extract_icl_set_of_hadm_ids(original_discharge_summaries, n=5):
    """Extract the last n admission ids as a list"""
    return list(original_discharge_summaries.tail(n)["hadm_id"])


def open_generated_summary(task, hadm_id, model, variation):
    """
    Load the generated summary for a document
    """
    target_file = (
        f"{RESULTS_DIR}/{model}/{variation}/{task}/{hadm_id}-discharge-inputs.txt"
    )
    with open(target_file, "r") as f:
        return f.read()


def open_target_summary(task, hadm_id):
    """
    Load the target (ground-truth) summary for a given task and admission id
    """
    # Baseline tasks uses re-identified summaries
    if "_baseline" in task:
        target_file = f"{RE_ID_TARGETS_ROOT}/{task}/{hadm_id}-target.txt"
    else:
        task = task.replace(f"{IN_CONTEXT_SUMMARY_TASK}", "")
        task = task.replace(f"{SANI_SUMM_SUMMARY_TASK}", "")
        target_file = f"{PSEUDO_TARGETS_ROOT}/{task}/{hadm_id}-target.txt"

    with open(target_file, "r") as f:
        return f.read()


def open_pseudonymized_summary(task, hadm_id):
    """
    Load the pseudonymized summary for a given hadm id
    """
    target_file = f"{PSEUDO_TARGETS_ROOT}/{task}/{hadm_id}-target.txt"
    with open(target_file, "r") as f:
        return f.read()


def fetch_admission_info(hadm_id):
    """Fetch admission info"""
    print(f"Admission info for {hadm_id}")
    ed_train_df = pd.read_csv(f"{TRAIN_DISCHARGE_ME}/edstays.csv")
    edstay = ed_train_df[ed_train_df["hadm_id"] == str(hadm_id)]
    return edstay


def run_packaging_for_colab():
    # create a tar of multiple directories using python
    import tarfile

    with tarfile.open("input-files-output.tar.gz", "a:") as target_pckg:
        target_pckg.add(f"{RE_ID_EXAMPLES_ROOT}/{SUMMARY_TYPES[0]}")
        target_pckg.add(f"{RE_ID_EXAMPLES_ROOT}/{SUMMARY_TYPES[1]}")
        target_pckg.add(f"{PSEUDO_TARGETS_ROOT}/{SUMMARY_TYPES[0]}")
        target_pckg.add(f"{PSEUDO_TARGETS_ROOT}/{SUMMARY_TYPES[1]}")


def result_file_is_present(task, hadm_id, target_model, variation_name) -> bool:
    """
    Check if the result file is present
    """
    target_file = f"{RESULTS_DIR}/{target_model}/{variation_name}/{task}/{hadm_id}-discharge-inputs.txt"
    return os.path.exists(target_file)


def reference_file_is_present(task, hadm_id) -> bool:
    """
    Check if the reference file is present
    """
    if "baseline" in task:
        return os.path.exists(f"{RE_ID_TARGETS_ROOT}/{task}/{hadm_id}-target.txt")
    else:
        task = task.replace(f"{IN_CONTEXT_SUMMARY_TASK}", "")
        task = task.replace(f"{SANI_SUMM_SUMMARY_TASK}", "")
        return os.path.exists(f"{PSEUDO_TARGETS_ROOT}/{task}/{hadm_id}-target.txt")


def open_legal_data():
    """
    Open the legal contracts data
    """
    print("Loading legal docs")
    with open(f"{LEGAL_EXAMPLES_ROOT}tldrlegal_v1.json", "r") as f:
        data = json.load(f)
        legal_data = {}
        for uid, values in data.items():
            legal_data[uid] = {
                "uid": values["uid"],
                "id": values["id"],
                "dataset": "tldrlegal_v1",
                "document": values["original_text"],
                "target": values["reference_summary"],
                "title": values["title"],
            }
    return legal_data


def get_cnn_reference_summary(summac_datapoint: dict, CNNDM_test: dict) -> str:
    """gets the target sumamry from the data"""

    _, _, id = summac_datapoint["cnndm_id"].split("-")

    return CNNDM_test[id]["highlights"]


def open_cnn_data():
    """
    Open the CNN/DailyMail data
    """
    print("Loading CNN docs")
    cnn_dataset = load_cnn_dataset()
    cnn_test = {v["id"]: v for v in cnn_dataset["test"]}

    data = {}
    for uid, values in cnn_test.items():
        data[uid] = {
            "id": values["id"],
            "dataset": "cnn_dailmail",
            "document": values["article"],
            "target": values["highlights"],
        }
    return data


def fetch_example(task):
    files = os.listdir(f"{ICL_EXAMPLES_ROOT}/{task}")
    random_file = random.choice(files)
    with open(f"{ICL_EXAMPLES_ROOT}/{task}/{random_file}", "r") as f:
        return f.read()


def store_utility_results(results, target_model, results_type):
    """Store results"""
    if os.path.exists(f"{UTILITY_RESULTS_DIR}/{results_type}") == False:
        os.makedirs(f"{UTILITY_RESULTS_DIR}/{results_type}")
    with open(f"{UTILITY_RESULTS_DIR}/{results_type}/{target_model}.json", "w") as f:
        json.dump(results, f, indent=4)


def store_privacy_results(results, target_model, results_type):
    """Store results"""
    if os.path.exists(f"{PRIVACY_RESULTS_DIR}/{results_type}") == False:
        os.makedirs(f"{PRIVACY_RESULTS_DIR}/{results_type}")
    with open(f"{PRIVACY_RESULTS_DIR}/{results_type}/{target_model}.json", "w") as f:
        json.dump(results, f, indent=4)


def read_file(filename):
    if not os.path.exists(filename):
        print(f"File {filename} does not exist")
        return None
    try:
        with open(filename, "r") as f:
            return f.read()
    except:
        print(f"Error reading file {filename}")
        return None


def write_to_file(filename, text):
    with open(filename, "w") as f:
        f.write(text)
