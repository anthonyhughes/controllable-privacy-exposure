# code reused from https://github.com/casszhao/PruneHall

import json
import nltk
from nltk import word_tokenize
import numpy as np
from constants import SUMMARY_TYPES, EXAMPLES_ROOT

data_stats = {}


def run(task):

    fname = f"{EXAMPLES_ROOT}legal_court/tldrlegal_v1.json"

    with open(fname, "r", encoding="utf8") as f:
        data = json.load(f)

    word_lengths = np.zeros(len(data))
    reference_lengths = np.zeros(len(data))

    for i, (key, value) in enumerate(data.items()):

        reference = (
            value.get("reference_summary", None)
            or value.get("claim", None)
            or value.get("target", None)
        )

        word_lengths[i] = len(word_tokenize(value["original_text"]))
        if reference is not None:
            reference_lengths[i] = len(word_tokenize(reference))

    data_stats[task] = {
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
    run("legal_court")
