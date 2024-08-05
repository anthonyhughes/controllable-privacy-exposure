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
    for task in SUMMARY_TYPES[0:1]:
        pii_property_counts = {}
        doc_count = 0
        token_count = 0
        for hadm_id in hadm_ids:
            result = run_pii_check(hadm_id, task, target_model)
            print(result.decode_mapping)
            pii_property_counts = update_pii_property_counts(
                result, pii_property_counts
            )
            counts = fetch_total_pii_count(result)
            if counts > 0:
                doc_count += 1
                token_count += counts
        print(f"Total documents with PII: {doc_count}")
        print(f"Total documents checked: {len(hadm_ids)}")
        print(f"PII percentage: {doc_count/len(hadm_ids)}")
        print(f"Total PII tokens: {token_count}")
        print(f"PII property counts: {pii_property_counts}")


if __name__ == "__main__":
    original_discharge_summaries = load_original_discharge_summaries()
    target_admission_ids = extract_hadm_ids(
        original_discharge_summaries=original_discharge_summaries, n=100
    )
    target_model = MODELS[1]
    target_admission_ids = [
        20731670,
        20783870,
        20783870,
        20280072,
        20356134,
        20237862,
        20285402,
    ]
    run_privacy_eval(hadm_ids=target_admission_ids, target_model=target_model)
