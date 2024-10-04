import json
from os import walk
import os
import time

from constants import (
    DISCHARGE_INSTRUCTIONS,
    FINAL_RAW_PRIVACY_RESULTS_DIR,
    IN_CONTEXT_SUMMARY_TASK,
    RE_ID_EXAMPLES_ROOT,
    BRIEF_HOSPITAL_COURSE,
    SANI_SUMM_SUMMARY_TASK,
)
from utils.reid_eval import get_all_profiles


def get_all_reidentified_documents_for_task(task=BRIEF_HOSPITAL_COURSE):
    target_file_path = f"{RE_ID_EXAMPLES_ROOT}/{task}/"
    return next(walk(target_file_path), (None, None, []))[2]


def build_mappings_for_task():
    """
    Run mapping
    """
    start_time = time.time()

    # Load all profiles once
    profiles = get_all_profiles()
    print("Loaded all profiles")

    # Load all target files once into memory
    target_files = get_all_reidentified_documents_for_task()
    print("Loaded all reidentified documents")

    # Read and store all target file contents in memory for faster access
    file_contents = {}
    for file in target_files:
        with open(f"{RE_ID_EXAMPLES_ROOT}/{BRIEF_HOSPITAL_COURSE}/{file}", "r") as f:
            file_contents[file] = "\n".join(
                f.readlines()[0:5]
            )  # only read the first 5 lines

    # Create a dictionary to store hadm_id to profile mappings
    hadm_id_to_profile_mapping = {}

    # Process each profile
    for i, pf in enumerate(profiles):
        name = pf["name"]
        print(f"Processing {i + 1}/{len(profiles)}")

        # Check each file content for the profile name
        for file, content in file_contents.items():
            if name in content:
                hadm_id = file.split("-")[0]  # Extract hadm_id
                hadm_id_to_profile_mapping[hadm_id] = pf

    print("Built the mapping")
    print(len(hadm_id_to_profile_mapping.keys()))

    # Save the results to a JSON file
    output_file = os.path.join(FINAL_RAW_PRIVACY_RESULTS_DIR, "profile_mappings.json")
    with open(output_file, "w") as f:
        json.dump(hadm_id_to_profile_mapping, f, indent=4)
    print(f"Saved {output_file}")

    end_time = time.time() - start_time
    print(f"Time taken: {end_time:.2f} seconds")


def is_token_in_another_profile(id, mappings, potential_leaked_token):
    c_mappings = mappings.copy()
    del c_mappings[id]
    flattened_names = [
        val
        for profile in c_mappings.values()
        for prop, val in profile.items()
        if prop in ["name", "clinician_name"]
    ]
    return potential_leaked_token in flattened_names


def run_reidentification_eval_v2(target_privacy_file):
    """
    Run the reidentification evaluation
    """
    # build_mappings_for_task()
    print
    output_file = os.path.join(FINAL_RAW_PRIVACY_RESULTS_DIR, "profile_mappings.json")
    with open(output_file, "r") as f:
        mappings = json.load(f)
    print("Loaded mappings")
    raw_privacy_results = os.path.join(
        FINAL_RAW_PRIVACY_RESULTS_DIR, target_privacy_file
    )
    with open(raw_privacy_results, "r") as f:
        privacy_results = json.load(f)

    task = BRIEF_HOSPITAL_COURSE
    v_name = "variation_1"

    for task in [
        BRIEF_HOSPITAL_COURSE, f"{BRIEF_HOSPITAL_COURSE}{IN_CONTEXT_SUMMARY_TASK}", f"{BRIEF_HOSPITAL_COURSE}{SANI_SUMM_SUMMARY_TASK}",
        DISCHARGE_INSTRUCTIONS, f"{DISCHARGE_INSTRUCTIONS}{IN_CONTEXT_SUMMARY_TASK}", f"{DISCHARGE_INSTRUCTIONS}{SANI_SUMM_SUMMARY_TASK}"
        ]:    
        print(f"Running reidentification eval for {task}")
        task_data = privacy_results[task]
        variation_data = task_data[v_name]
        hadm_ids = variation_data.keys()
        person = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        # dates = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        # orgs = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        # all ids of patients in the generated summaries
        for id in hadm_ids:
            if variation_data[id]["counts"]["PERSON"] > 0:
                leaked_tokens = variation_data[id]["sanitized_encodings"]
                if id not in mappings:
                    continue
                matched_profile = mappings[id]
                matched_profile = {
                    k: v for k, v in matched_profile.items() if k in ["name"]
                }
                for potential_leaked_token, token_type in leaked_tokens.items():
                    # is the key an item in the matched profile
                    if "PERSON" in token_type:
                        for _, profile_v in matched_profile.items():
                            potential_leaked_token = str(potential_leaked_token)
                            if potential_leaked_token in profile_v:
                                person["tp"] += 1
                            elif (
                                potential_leaked_token not in profile_v
                                and is_token_in_another_profile(
                                    id, mappings, potential_leaked_token
                                )
                            ):
                                person["fp"] += 1
                            elif potential_leaked_token not in profile_v:
                                person["fn"] += 1
        print('Person', person)
        # calc precision and recall
        try:
            precision = person["tp"] / (person["tp"] + person["fp"])
        except ZeroDivisionError:
            precision = 0
        try:
            recall = person["tp"] / (person["tp"] + person["fn"])
        except ZeroDivisionError:
            recall = 0
        print('Precision', precision)
        print('Recall', recall)
        print(f'Done {task}')