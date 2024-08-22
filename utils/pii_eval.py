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


def create_average_pii_per_property(pii_property_counts, doc_count):
    """
    Create average PII per property
    """
    average_pii_per_property = {}
    for entity in pii_property_counts.keys():
        average_pii_per_property[entity] = pii_property_counts[entity] / doc_count
    return average_pii_per_property


def print_as_latex_table(results, model):
    for key in results.keys():
        res = results[key]
        exposed_res = res["exposed_pii_per_property"]
        props_to_keep = ["PERSON", "DATE", "ORG", "PERSON", "LOC"]
        exposed_res = {k: exposed_res[k] for k in props_to_keep}
        exposed_res = dict(sorted(exposed_res.items()))
        exposed_res["total"] = sum(exposed_res.values())
        exposed_res["latex"] = " & ".join([str(v) for v in exposed_res.values()])
        with(open(f"data/results/privacy_latex.txt", "a")) as f:
            f.write(f"{key}-{model} & {exposed_res['latex']} \\\\ \n")


def run_privacy_eval(hadm_ids, target_model):
    """
    Run PII evaluation for all documents
    """
    results = {}
    raw_results = {}
    for task in SUMMARY_TYPES:
        all_token_counts = []
        baseline_all_token_counts = []
        all_normalised_token_counts = []
        baseline_normalised_token_counts = []
        pii_property_counts = {}
        baseline_pii_property_counts = {}
        for hadm_id in hadm_ids:
            result = run_pii_check(hadm_id, task, target_model)
            token_length = len(result.text.split())
            pii_property_counts = update_pii_property_counts(
                result, pii_property_counts
            )
            counts = fetch_total_pii_count(result)
            all_token_counts.append(counts)
            all_normalised_token_counts.append(counts / token_length)

            baseline_task = f"{task}_baseline"
            baseline_results = run_pii_check(hadm_id, baseline_task, target_model)
            baseline_pii_property_counts = update_pii_property_counts(
                baseline_results, baseline_pii_property_counts
            )
            baseline_counts = fetch_total_pii_count(baseline_results)
            baseline_all_token_counts.append(baseline_counts)
            baseline_normalised_token_counts.append(baseline_counts / token_length)
        doc_count = len(list(filter(lambda x: x > 0, all_token_counts)))
        baseline_doc_count = len(
            list(filter(lambda x: x > 0, baseline_all_token_counts))
        )
        results[task] = {
            "exposed_docs_count": doc_count,
            "docs_count": len(hadm_ids),
            "exposed_tokens_count": sum(all_token_counts),
            "normalised_exposed_tokens_count": sum(all_normalised_token_counts)
            / len(all_normalised_token_counts),
            "exposed_pii_per_property": pii_property_counts,
            "average_exposed_pii_per_property_per_document": create_average_pii_per_property(
                pii_property_counts, doc_count
            ),
            "pii__document_percentage": doc_count / len(hadm_ids),
        }
        results[baseline_task] = {
            "exposed_docs_count": len(
                list(filter(lambda x: x > 0, baseline_all_token_counts))
            ),
            "docs_count": len(hadm_ids),
            "exposed_tokens_count": sum(baseline_all_token_counts),
            "normalised_exposed_tokens_count": sum(baseline_normalised_token_counts)
            / len(baseline_normalised_token_counts),
            "exposed_pii_per_property": baseline_pii_property_counts,
            "average_exposed_pii_per_property_per_document": create_average_pii_per_property(
                baseline_pii_property_counts, baseline_doc_count
            ),
            "pii__document_percentage": baseline_doc_count / len(hadm_ids),
        }
    store_results(results, target_model, "privacy")
    store_results(raw_results, target_model, "raw_privacy")
    print_as_latex_table(results, target_model)


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
