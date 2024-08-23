import os
from datasets import load_dataset
from timeit import default_timer as timer

import pandas as pd

from constants import (
    RE_ID_EXAMPLES_ROOT,
    RESULTS_DIR,
    PSEUDO_TARGETS_ROOT,
    SUMMARY_TYPES,
    TRAIN_DISCHARGE_ME,
    RE_ID_TARGETS_ROOT,
)


# load a huggingface dataset
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
    return list(original_discharge_summaries.head(n)["hadm_id"])


def open_generated_summary(task, hadm_id, model):
    """
    Load the generated summary for a document
    """
    if "baseline" in task and model == "gpt-4o-mini":
        task = task.replace("_summary_task", "")
        target_file = (
            f"{RESULTS_DIR}/{model}/{task}/{hadm_id}_{task}_summary_task_summary.txt"
        )
    elif "baseline" not in task and model == "gpt-4o-mini":
        target_file = f"{RESULTS_DIR}/{model}/{task}/{hadm_id}_{task}_summary.txt"
    else:
        task = task.replace("_summary_task", "")
        target_file = f"{RESULTS_DIR}/{model}/{task}/{hadm_id}-discharge-inputs.txt"

    with open(target_file, "r") as f:
        return f.read()


def open_target_summary(task, hadm_id):
    """
    Load the target summary for a document
    """
    if "baseline" in task:
        target_file = f"{RE_ID_TARGETS_ROOT}/{task}/{hadm_id}-target.txt"
    else:
        target_file = f"{RE_ID_EXAMPLES_ROOT}/{task}/{hadm_id}-discharge-inputs.txt"
    with open(target_file, "r") as f:
        return f.read()


def fetch_admission_info(hadm_id):
    """Fetch admission info"""
    print(f"Admission info for {hadm_id}")
    df = pd.read_csv(f"{TRAIN_DISCHARGE_ME}/edstays.csv")
    edstay = df[df["hadm_id"] == str(hadm_id)]
    return edstay


def run_packaging_for_colab():
    # create a tar of multiple directories using python
    import tarfile

    with tarfile.open("input-files-output.tar.gz", "a:") as target_pckg:
        target_pckg.add(f"{RE_ID_EXAMPLES_ROOT}/{SUMMARY_TYPES[0]}")
        target_pckg.add(f"{RE_ID_EXAMPLES_ROOT}/{SUMMARY_TYPES[1]}")
        target_pckg.add(f"{PSEUDO_TARGETS_ROOT}/{SUMMARY_TYPES[0]}")
        target_pckg.add(f"{PSEUDO_TARGETS_ROOT}/{SUMMARY_TYPES[1]}")


def result_file_is_present(task, hadm_id, target_model) -> bool:
    """
    Check if the result file is present
    """
    if "baseline" in task and target_model == "gpt-4o-mini":
        task = task.replace("_summary_task", "")
        target_file = f"{RESULTS_DIR}/{target_model}/{task}/{hadm_id}_{task}_summary_task_summary.txt"
        return os.path.exists(target_file)
    elif "baseline" not in task and target_model == "gpt-4o-mini":
        target_file = (
            f"{RESULTS_DIR}/{target_model}/{task}/{hadm_id}_{task}_summary.txt"
        )
        return os.path.exists(target_file)
    else:
        task = task.replace("_summary_task", "")
        target_file = (
            f"{RESULTS_DIR}/{target_model}/{task}/{hadm_id}-discharge-inputs.txt"
        )
        return os.path.exists(target_file)


if __name__ == "__main__":
    start = timer()
    run_packaging_for_colab()
    end = timer() - start
    print(f"Time to complete in secs: {end}")
