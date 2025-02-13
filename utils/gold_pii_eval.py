import json
import os
from pydeidentify import Deidentifier


def fetch_total_pii_count(scrubber_result):
    """
    Check if the text contains PII
    """
    token_count = 0
    pii_counts = scrubber_result.counts
    for entity in pii_counts.keys():
        pii_counts[entity] >= 1
        token_count += pii_counts[entity]

    return token_count


def update_original_raw_results(raw_results, result, model, f_id, task_type):
    """Capture all de-id results"""
    if model not in raw_results[task_type]:
        raw_results[task_type][model] = {}

    raw_results[task_type][model].update(
        {
            f_id: {
                "sanitized_encodings": result.encode_mapping,
                "sanitized_text": result.text,
                "counts": result.counts,
            }
        },
    )
    return raw_results


def save_priv_result(result):
    with open(f"data/gold/privacy_results.json", "w") as f:
        f.write(json.dumps(result, indent=4))


def run_gold_privacy_eval():
    """
    Run PII evaluation for all documents
    """
    d = Deidentifier()
    raw_input_results = {
        "reid": {},
        "scrubbed": {},
    }
    models = ["gpt-4o-mini", "claude-3-5-sonnet-20240620"]
    for model in models:
        for task_type in ["reid", "scrubbed"]:
            files = os.listdir(f"data/gold/{model}/{task_type}")
            for f_id in files:
                with open(f"data/gold/{model}/{task_type}/{f_id}", "r") as f:
                    content = f.read()
                    scrubbed_text = d.deidentify(content)
                    raw_input_results = update_original_raw_results(
                        raw_input_results, scrubbed_text, model, f_id, task_type
                    )
    save_priv_result(raw_input_results)
    print("Done Task")
    print("Done")


if __name__ == "__main__":
    run_gold_privacy_eval()
