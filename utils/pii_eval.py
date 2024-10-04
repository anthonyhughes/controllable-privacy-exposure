from datetime import datetime
import os
from constants import (
    BASELINE_SUMMARY_TASK,
    IN_CONTEXT_SUMMARY_TASK,
    MODELS,
    PRIVACY_RESULTS_DIR,
    SANI_SUMM_SUMMARY_TASK,
    SUMMARY_TYPES,
    TASK_SUFFIXES,
)

from mimic.mimic_data import load_original_discharge_summaries
from utils.dataset_utils import (
    extract_hadm_ids,
    extract_hadm_ids_from_dir,
    open_generated_summary,
    result_file_is_present,
    store_privacy_results,
)
from pydeidentify import Deidentifier
from utils.prompt_variations import variations

d = Deidentifier()


def run_pii_check(hadm_id, task, target_model, variation):
    """
    Run PII evaluation for a single document
    """
    generated_summary = open_generated_summary(task, hadm_id, target_model, variation)
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
        if doc_count == 0:
            average_pii_per_property[entity] = 0
        else:
            average_pii_per_property[entity] = pii_property_counts[entity] / doc_count
    return average_pii_per_property


def print_as_latex_table(results, model):
    """
    Print the results as a latex table
    """
    for key in results.keys():
        res = results[key]
        exposed_res = res["exposed_pii_per_property"]
        props_to_keep = ["PERSON", "DATE", "ORG", "PERSON", "LOC"]
        exposed_res = {k: exposed_res[k] for k in props_to_keep}
        exposed_res = dict(sorted(exposed_res.items()))
        exposed_res["total"] = sum(exposed_res.values())
        exposed_res["latex"] = " & ".join([str(v) for v in exposed_res.values()])
        if os.path.exists(f"{PRIVACY_RESULTS_DIR}/latex") == False:
            os.makedirs(f"{PRIVACY_RESULTS_DIR}/latex")
        with open(f"{PRIVACY_RESULTS_DIR}/latex/privacy.txt", "a") as f:
            f.write(f"{datetime.now()} \\\\ \n")
            f.write(f"{key}-{model} & {exposed_res['latex']} \\\\ \n")


def update_raw_results(raw_results, result, task, hadm_id, variation):
    """Capture all de-id results"""

    # Capture potential updates
    if task not in raw_results:
        raw_results[task] = {}
    if variation not in raw_results[task]:
        raw_results[task][variation] = {}

    raw_results[task][variation].update(
        {
            hadm_id: {
                "sanitized_encodings": result.encode_mapping,
                "sanitized_text": result.text,
                "counts": result.counts,
            }
        },
    )
    return raw_results


def update_overall_pii_results(
    sub_task,
    hadm_id,
    target_model,
    variation,
    pii_property_counts,
    all_token_counts,
    normalised_token_counts,
    token_length,
    raw_results,
):
    if result_file_is_present(sub_task, hadm_id, target_model, variation):
        sani_summ_results = run_pii_check(hadm_id, sub_task, target_model, variation)
        raw_results = update_raw_results(
            raw_results, sani_summ_results, sub_task, hadm_id, variation
        )
        pii_property_counts = update_pii_property_counts(
            sani_summ_results, pii_property_counts
        )
        sani_summ_counts = fetch_total_pii_count(sani_summ_results)
        all_token_counts.append(sani_summ_counts)
        try:
            ntc = sani_summ_counts / token_length
        except ZeroDivisionError:
            ntc = 0
        normalised_token_counts.append(ntc)
        return pii_property_counts, all_token_counts, normalised_token_counts


def populate_missing_props(results, task, variation):
    if task not in results:
        results[task] = {}
    if variation not in results[task]:
        results[task][variation] = {}
    return results


def update_results_for_task(
    results,
    task,
    variation,
    leaked_doc_count,
    hadm_ids,
    token_counts,
    normalised_token_counts,
    pii_property_counts,
):
    """
    Update the results for a task
    """
    try:
        avg_normalised_token_counts = sum(normalised_token_counts) / len(normalised_token_counts),
    except ZeroDivisionError:
        avg_normalised_token_counts = 0

    try:
        avg_document_percentage = leaked_doc_count / len(hadm_ids)
    except ZeroDivisionError:
        avg_document_percentage = 0

    results[task][variation] = {
        "exposed_docs_count": leaked_doc_count,
        "docs_count": len(hadm_ids),
        "exposed_tokens_count": sum(token_counts),
        "normalised_exposed_tokens_count": avg_normalised_token_counts,
        "exposed_pii_per_property": pii_property_counts,
        "average_exposed_pii_per_property_per_document": create_average_pii_per_property(
            pii_property_counts, leaked_doc_count
        ),
        "pii_document_percentage": avg_document_percentage,
    }
    return results


def run_privacy_eval(target_model, tasks=SUMMARY_TYPES, sub_tasks=TASK_SUFFIXES):
    """
    Run PII evaluation for all documents
    """
    results = {}
    raw_results = {}
    for task in tasks:
        for variation_prompt in variations:
            variation = variation_prompt["name"]
            baseline_all_token_counts = []
            all_token_counts = []
            icl_all_token_counts = []
            sani_summ_all_token_counts = []

            baseline_normalised_token_counts = []
            all_normalised_token_counts = []
            icl_normalised_token_counts = []
            sani_summ_normalised_token_counts = []

            baseline_pii_property_counts = {}
            pii_property_counts = {}
            icl_pii_property_counts = {}
            sani_summ_pii_property_counts = {}

            hadm_ids = extract_hadm_ids_from_dir(target_model, task, variation)
            for i, hadm_id in enumerate(hadm_ids):
                # run privacy evaluation for privsumm
                if "" in sub_tasks and result_file_is_present(
                    task, hadm_id, target_model, variation
                ):
                    print(
                        f"Running PII evaluation for model, {target_model}, task {task}, variation {variation}, on id {hadm_id} - {i+1}/{len(hadm_ids)}"
                    )
                    priv_result = run_pii_check(hadm_id, task, target_model, variation)
                    raw_results = update_raw_results(
                        raw_results, priv_result, task, hadm_id, variation
                    )
                    pii_property_counts = update_pii_property_counts(
                        priv_result, pii_property_counts
                    )
                    counts = fetch_total_pii_count(priv_result)
                    all_token_counts.append(counts)

                    priv_token_length = len(priv_result.text.split())
                    all_normalised_token_counts.append(counts / priv_token_length)

                # run privacy evaluation for baseline
                baseline_task = f"{task}{BASELINE_SUMMARY_TASK}"
                if BASELINE_SUMMARY_TASK in sub_tasks and result_file_is_present(
                    baseline_task, hadm_id, target_model, variation
                ):
                    print(
                        f"Running PII evaluation for task {baseline_task}, variation {variation}, on id {hadm_id} - {i+1}/{len(hadm_ids)}"
                    )
                    baseline_results = run_pii_check(
                        hadm_id, baseline_task, target_model, variation
                    )
                    raw_results = update_raw_results(
                        raw_results, baseline_results, baseline_task, hadm_id, variation
                    )
                    baseline_pii_property_counts = update_pii_property_counts(
                        baseline_results, baseline_pii_property_counts
                    )
                    baseline_counts = fetch_total_pii_count(baseline_results)
                    baseline_all_token_counts.append(baseline_counts)

                    baseline_token_length = len(baseline_results.text.split())
                    baseline_normalised_token_counts.append(
                        baseline_counts / baseline_token_length
                    )

                # run privacy evaluation for icl-privsumm
                icl_task = f"{task}{IN_CONTEXT_SUMMARY_TASK}"
                if IN_CONTEXT_SUMMARY_TASK in sub_tasks and result_file_is_present(
                    icl_task, hadm_id, target_model, variation
                ):
                    print(
                        f"Running PII evaluation for task {icl_task}, variation {variation}, on id {hadm_id} - {i+1}/{len(hadm_ids)}"
                    )
                    icl_results = run_pii_check(
                        hadm_id, icl_task, target_model, variation
                    )
                    raw_results = update_raw_results(
                        raw_results, icl_results, icl_task, hadm_id, variation
                    )
                    icl_pii_property_counts = update_pii_property_counts(
                        icl_results, icl_pii_property_counts
                    )
                    icl_counts = fetch_total_pii_count(icl_results)
                    icl_all_token_counts.append(icl_counts)

                    icl_token_length = len(icl_results.text.split())
                    icl_normalised_token_counts.append(icl_counts / icl_token_length)

                sani_summ_task = f"{task}{SANI_SUMM_SUMMARY_TASK}"
                print(
                    f"Running PII evaluation for task {sani_summ_task}, variation {variation}, on id {hadm_id} - {i+1}/{len(hadm_ids)}"
                )
                if SANI_SUMM_SUMMARY_TASK in sub_tasks and result_file_is_present(
                    sani_summ_task, hadm_id, target_model, variation
                ):
                    print(
                        f"Running PII evaluation for task {sani_summ_task}, variation {variation}, on id {hadm_id} - {i+1}/{len(hadm_ids)}"
                    )
                    sani_summ_results = run_pii_check(
                        hadm_id, sani_summ_task, target_model, variation
                    )
                    raw_results = update_raw_results(
                        raw_results,
                        sani_summ_results,
                        sani_summ_task,
                        hadm_id,
                        variation,
                    )
                    sani_summ_pii_property_counts = update_pii_property_counts(
                        sani_summ_results, sani_summ_pii_property_counts
                    )
                    sani_summ_counts = fetch_total_pii_count(sani_summ_results)
                    sani_summ_all_token_counts.append(sani_summ_counts)

                    sani_summ_token_length = len(sani_summ_results.text.split())
                    try:
                        nrom_res = sani_summ_counts / sani_summ_token_length
                    except ZeroDivisionError:
                        nrom_res = 0
                    
                    sani_summ_normalised_token_counts.append(
                        nrom_res
                    )

            # Take account of all docs with a leaked token
            doc_count = len(list(filter(lambda x: x > 0, all_token_counts)))

            # Take account of all docs with a leaked token in the baseline summaries
            baseline_doc_count = len(
                list(filter(lambda x: x > 0, baseline_all_token_counts))
            )
            # Take account of all docs with a leaked token in the ICL
            icl_doc_count = len(list(filter(lambda x: x > 0, icl_all_token_counts)))

            # Take account of all docs with a leaked token in the SaniSumm
            sani_summ_doc_count = len(
                list(filter(lambda x: x > 0, sani_summ_all_token_counts))
            )

            for sub_task in sub_tasks:
                full_sub_task = f"{task}{sub_task}"
                results = populate_missing_props(
                    results, full_sub_task, variation=variation
                )

            if "" in sub_tasks:
                results = update_results_for_task(
                    results,
                    task,
                    variation,
                    doc_count,
                    hadm_ids,
                    all_token_counts,
                    all_normalised_token_counts,
                    pii_property_counts,
                )
            if BASELINE_SUMMARY_TASK in sub_tasks:
                results = update_results_for_task(
                    results,
                    baseline_task,
                    variation,
                    baseline_doc_count,
                    hadm_ids,
                    baseline_all_token_counts,
                    baseline_normalised_token_counts,
                    baseline_pii_property_counts,
                )
            if IN_CONTEXT_SUMMARY_TASK in sub_tasks:
                results = update_results_for_task(
                    results,
                    icl_task,
                    variation,
                    icl_doc_count,
                    hadm_ids,
                    icl_all_token_counts,
                    icl_normalised_token_counts,
                    icl_pii_property_counts,
                )
            if SANI_SUMM_SUMMARY_TASK in sub_tasks:
                results = update_results_for_task(
                    results,
                    sani_summ_task,
                    variation,
                    sani_summ_doc_count,
                    hadm_ids,
                    sani_summ_all_token_counts,
                    sani_summ_normalised_token_counts,
                    sani_summ_pii_property_counts,
                )

    store_privacy_results(raw_results, target_model, f"final_raw_privacy")
    store_privacy_results(results, target_model, f"final_privacy")
    print("Done Task")
    print("Done")

if __name__ == "__main__":
    original_discharge_summaries = load_original_discharge_summaries()
    target_admission_ids = extract_hadm_ids(
        original_discharge_summaries=original_discharge_summaries, n=100
    )
    for model in MODELS:
        run_privacy_eval(hadm_ids=target_admission_ids, target_model=model)
