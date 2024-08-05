from datasets import load_dataset
from timeit import default_timer as timer

from constants import RESULTS_DIR, PSEUDO_TARGETS_ROOT


# load a huggingface dataset
def load_cnn_dataset():
    """"""
    dataset = load_dataset("ccdv/cnn_dailymail", "3.0.0", trust_remote_code=True)
    return dataset


def update_article_prefix(dataset, prefix):
    """Update the article prefix"""
    dataset = dataset.map(lambda example: {"article": prefix + example["article"]})
    return dataset


def tester():
    dataset = load_cnn_dataset()
    somerow = dataset["train"][0]
    print(somerow["id"])
    print(somerow["article"][0:100])
    print(somerow["highlights"])
    updated_dataset = update_article_prefix(
        dataset, "You are an expert is news. Summarise the following article:\n"
    )
    somerow = updated_dataset["train"][0]
    print(somerow["id"])
    print(somerow["article"][0:100])
    print(somerow["highlights"])


def extract_hadm_ids(original_discharge_summaries, n=100):
    """Extract the first n admission ids as a list"""
    return list(original_discharge_summaries.head(n)["hadm_id"])


def open_generated_summary(task, hadm_id, model):
    """
    Load the generated summary for a document
    """
    with open(f"{RESULTS_DIR}/{model}/{task}/{hadm_id}_{task}_summary.txt", "r") as f:
        return f.read()


def open_target_summary(task, hadm_id):
    """
    Load the target summary for a document
    """
    with open(f"{PSEUDO_TARGETS_ROOT}{task}/{hadm_id}-target.txt", "r") as f:
        return f.read()


if __name__ == "__main__":
    start = timer()
    tester()
    end = timer() - start
    print(f"Time to complete in secs: {end}")
