import json
from os import walk
import os
import time

from constants import (
    DISCHARGE_INSTRUCTIONS,
    FINAL_RAW_PRIVACY_RESULTS_DIR,
    FINAL_REID_RESULTS_DIR,
    IN_CONTEXT_SUMMARY_TASK,
    RE_ID_EXAMPLES_ROOT,
    BRIEF_HOSPITAL_COURSE,
    SANI_SUMM_SUMMARY_TASK,
    TASK_SUFFIXES,
)
from utils.reid_eval import get_all_profiles


def get_all_reidentified_documents_for_task(task=BRIEF_HOSPITAL_COURSE):
    target_file_path = f"{RE_ID_EXAMPLES_ROOT}/{task}/"
    return next(walk(target_file_path), (None, None, []))[2]


def build_mappings_for_task(task):
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
        with open(f"{RE_ID_EXAMPLES_ROOT}/{task}/{file}", "r") as f:
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
                for prop in [
                    "in_date",
                    "out_date",
                    "intervention_date",
                    "birth_date",
                    "clinician_name",
                ]:
                    if prop == "clinician_name":
                        res = pf[prop].split("Dr. ")
                        if len(res) > 1:
                            pf[prop] = res[1]
                    else:
                        pf[prop] = pf[prop].split(" ")[0]
                hadm_id_to_profile_mapping[hadm_id] = pf
        if len(hadm_id_to_profile_mapping) == len(file_contents):
            break
    print("Built the mapping")
    print(len(hadm_id_to_profile_mapping.keys()))

    # Save the results to a JSON file
    output_file = os.path.join(
        FINAL_RAW_PRIVACY_RESULTS_DIR, f"{task}_profile_mappings.json"
    )
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


def run_reidentification_eval_v2(target_privacy_file, tasks, model):
    """
    Run the reidentification evaluation
    """
    raw_privacy_results = os.path.join(
        FINAL_RAW_PRIVACY_RESULTS_DIR, target_privacy_file
    )
    with open(raw_privacy_results, "r") as f:
        privacy_results = json.load(f)
    final_results = {}
    for task in tasks:
        # build_mappings_for_task(task)
        output_file = os.path.join(
            FINAL_RAW_PRIVACY_RESULTS_DIR, f"{task}_profile_mappings.json"
        )
        print(f"Loading mappings from {output_file}")
        with open(output_file, "r") as f:
            mappings = json.load(f)
        print("Loaded mappings")

        v_name = "variation_1"

        for sub_task_suffix in TASK_SUFFIXES[1:4]:
            final_results[f"{task}{sub_task_suffix}"] = {}
            print(f"Running reidentification eval for {task}{sub_task_suffix}")
            sub_task = f"{task}{sub_task_suffix}"
            print(f"Running reidentification eval for {sub_task}")
            task_data = privacy_results[sub_task]
            variation_data = task_data[v_name]
            hadm_ids = variation_data.keys()
            conf_look_up = {
                "PERSON": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
                "DATE": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
                "ORG": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
            }
            # all ids of patients in the generated summaries
            for id in hadm_ids:
                prop_lookup = {
                    "PERSON": ["name", "clinician_name"],
                    "DATE": ["in_date", "out_date", "intervention_date", "birth_date"],
                    "ORG": ["location"],
                }
                for property in ["PERSON", "DATE", "ORG"]:
                    if variation_data[id]["counts"][property] > 0:
                        leaked_tokens = variation_data[id]["sanitized_encodings"]
                        if id not in mappings:
                            continue
                        matched_profile = mappings[id]
                        matched_profile = {
                            k: v
                            for k, v in matched_profile.items()
                            if k in prop_lookup[property]
                        }
                        for potential_leaked_token, token_type in leaked_tokens.items():
                            # is the key an item in the matched profile
                            if property in token_type:
                                for _, profile_v in matched_profile.items():
                                    potential_leaked_token = str(potential_leaked_token)
                                    if potential_leaked_token in profile_v:
                                        conf_look_up[property]["tp"] += 1
                                    elif (
                                        potential_leaked_token not in profile_v
                                        and is_token_in_another_profile(
                                            id, mappings, potential_leaked_token
                                        )
                                    ):
                                        conf_look_up[property]["fp"] += 1
                                    elif potential_leaked_token not in profile_v:
                                        conf_look_up[property]["fn"] += 1
            for property in ["PERSON", "DATE", "ORG"]:
                print(property, conf_look_up[property])
                # calc precision and recall
                try:
                    precision = conf_look_up[property]["tp"] / (
                        conf_look_up[property]["tp"] + conf_look_up[property]["fp"]
                    )
                    conf_look_up[property]["precision"] = precision
                except ZeroDivisionError:
                    conf_look_up[property]["precision"] = 0

                try:
                    recall = conf_look_up[property]["tp"] / (
                        conf_look_up[property]["tp"] + conf_look_up[property]["fn"]
                    )
                    conf_look_up[property]["recall"] = recall
                except ZeroDivisionError:
                    conf_look_up[property]["recall"] = 0

                # calc false positive rate
                try:
                    fpr = conf_look_up[property]["fp"] / (
                        conf_look_up[property]["fp"] + conf_look_up[property]["tn"]
                    )
                    conf_look_up[property]["fpr"] = fpr
                except ZeroDivisionError:
                    conf_look_up[property]["fpr"] = 0
                    
            final_results[f"{task}{sub_task_suffix}"] = conf_look_up
    cdatetime = time.strftime("%Y%m%d-%H%M%S")
    with open(
        os.path.join(FINAL_REID_RESULTS_DIR, f"{model}-reidentification_results-{cdatetime}.json"),
        "w",
    ) as f:
        json.dump(final_results, f, indent=4)
