import json
from constants import (
    MODELS,
    SUMMARY_TYPES,
)

from mimic.mimic_data import load_original_discharge_summaries
from utils.dataset_utils import extract_hadm_ids, open_generated_summary
from pydeidentify import Deidentifier

d = Deidentifier()


def run_pii_check(hadm_id, task, target_model):
    """
    Run PII evaluation for a single document
    """
    print(f"Running PII evaluation for {hadm_id}")
    print(f"Running PII evaluation for task: {task}")
    generated_summary = open_generated_summary(task, hadm_id, target_model)
    scrubbed_text = d.deidentify(generated_summary)
    return scrubbed_text


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


def update_pii_property_counts(scrubber_result, pii_property_counts):
    """
    Update the PII property counts
    """
    pii_counts = scrubber_result.counts
    for entity in pii_counts.keys():
        if entity not in pii_property_counts:
            pii_property_counts[entity] = 0
        if pii_counts[entity] >= 1:
            pii_property_counts[entity] += pii_counts[entity]
    return pii_property_counts


def run_privacy_eval(hadm_ids, target_model):
    """
    Run PII evaluation for all documents
    """
    results = {}
    for task in SUMMARY_TYPES:
        pii_property_counts = {}
        baseline_doc_count = 0
        doc_count = 0
        baseline_token_count = 0
        token_count = 0
        for hadm_id in hadm_ids:
            baseline_task = f"{task}_baseline_summary_task"
            baseline_results = run_pii_check(hadm_id, baseline_task, target_model)
            baseline_pii_property_counts = update_pii_property_counts(
                baseline_results, pii_property_counts
            )
            baseline_counts = fetch_total_pii_count(baseline_results)

            result = run_pii_check(hadm_id, task, target_model)
            pii_property_counts = update_pii_property_counts(
                result, pii_property_counts
            )
            counts = fetch_total_pii_count(result)

            # Build privacy counts
            if counts > 0:
                doc_count += 1
                token_count += counts
            if baseline_counts > 0:
                baseline_doc_count += 1
                baseline_token_count += baseline_counts
        results[task] = {
            "exposed_docs_count": doc_count,
            "docs_count": len(hadm_ids),
            "exposed_tokens_count": token_count,
            "exposed_pii_per_property": pii_property_counts,
            "pii__document_percentage": doc_count / len(hadm_ids),
        }
        results[baseline_task] = {
            "exposed_docs_count": baseline_doc_count,
            "docs_count": len(hadm_ids),
            "exposed_tokens_count": baseline_token_count,
            "exposed_pii_per_property": baseline_pii_property_counts,
            "pii__document_percentage": baseline_doc_count / len(hadm_ids),
        }
    store_results(results, target_model, "privacy")


def store_results(results, target_model, results_type):
    """Store results"""
    with open(f"data/results_{results_type}_{target_model}.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    original_discharge_summaries = load_original_discharge_summaries()
    target_admission_ids = extract_hadm_ids(
        original_discharge_summaries=original_discharge_summaries, n=100
    )
    for model in MODELS:
        run_privacy_eval(hadm_ids=target_admission_ids, target_model=model)
