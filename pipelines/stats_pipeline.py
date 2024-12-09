# code reused from https://github.com/casszhao/PruneHall

import json
import os
import nltk
from nltk import word_tokenize
import numpy as np
from constants import SUMMARY_TYPES, EXAMPLES_ROOT
from utils.dataset_utils import open_cnn_data

data_stats = {}


def run():
    # task = "legal_court"
    # fname = f"{EXAMPLES_ROOT}legal_court/tldrlegal_v1.json"

    # with open(fname, "r", encoding="utf8") as f:
    #     data = json.load(f)

    # word_lengths = np.zeros(len(data))
    # reference_lengths = np.zeros(len(data))

    # for i, (key, value) in enumerate(data.items()):

    #     reference = (
    #         value.get("reference_summary", None)
    #         or value.get("claim", None)
    #         or value.get("target", None)
    #     )

    #     word_lengths[i] = len(word_tokenize(value["original_text"]))
    #     if reference is not None:
    #         reference_lengths[i] = len(word_tokenize(reference))

    # data_stats[task] = {
    #     "dataset_size": len(data),
    #     "source_mean": round(word_lengths.mean(), 1),
    #     "source_max": round(word_lengths.max(), 1),
    #     "reference_mean": round(reference_lengths.mean(), 1),
    #     "reference_max": round(reference_lengths.max(), 1),
    # }


    # task = "discharge_instructions"
    # source_files = [file for file in os.listdir(f"{EXAMPLES_ROOT}{task}/") if file.endswith("-discharge-inputs.txt")]
    # reference_files = [file for file in os.listdir(f"{EXAMPLES_ROOT}{task}/") if file.endswith("-target.txt")]

    # word_lengths = np.zeros(len(source_files))
    # reference_lengths = np.zeros(len(reference_files))

    # for i, source_file in enumerate(source_files):
    #     with open(f"{EXAMPLES_ROOT}{task}/{source_file}", "r", encoding="utf8") as f:
    #         source_text = f.read()
    #     word_lengths[i] = len(word_tokenize(source_text))    

    # for i, reference_file in enumerate(reference_files):
    #     with open(f"{EXAMPLES_ROOT}{task}/{reference_file}", "r", encoding="utf8") as f:
    #         reference_text = f.read()
    #     reference_lengths[i] = len(word_tokenize(reference_text))

    # data_stats[task] = {
    #     "dataset_size": len(reference_files),
    #     # "source_mean": round(word_lengths.mean(), 1),
    #     # "source_max": round(word_lengths.max(), 1),
    #     "reference_mean": round(reference_lengths.mean(), 1),
    #     "reference_max": round(reference_lengths.max(), 1),
    # }

    data = open_cnn_data()
    word_lengths = np.zeros(len(data))
    reference_lengths = np.zeros(len(data))

    for i, (key, value) in enumerate(data.items()):

        reference = (
            value.get("reference_summary", None)
            or value.get("claim", None)
            or value.get("target", None)
        )

        word_lengths[i] = len(word_tokenize(value["document"]))
        if reference is not None:
            reference_lengths[i] = len(word_tokenize(reference))

    data_stats["cnn"] = {
        "dataset_size": len(data),
        "source_mean": round(word_lengths.mean(), 1),
        "source_max": round(word_lengths.max(), 1),
        "reference_mean": round(reference_lengths.mean(), 1),
        "reference_max": round(reference_lengths.max(), 1),
    }
    with open("data/stats.json", "w", encoding="utf8") as f:
        json.dump(data_stats, f, indent=4, default=str)


if __name__ == "__main__":
    nltk.download("punkt")
    run()
