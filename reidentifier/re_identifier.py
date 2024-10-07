import json
import os
import re
from constants import (
    BASELINE_SUMMARY_TASK,
    EXAMPLES_ROOT,
    RE_ID_EXAMPLES_ROOT,
    SUMMARY_TYPES,
    RE_ID_TARGETS_ROOT,
)
from mimic.mimic_data import load_original_discharge_summaries
from reidentifier.reidentify_utils import (
    fetch_file_names,
    generate_random_profile,
    load_file,
    generate_random_unit_number,
    remove_extra_redactions,
)
from utils.dataset_utils import extract_hadm_ids, reference_file_is_present


def fill_in_discharge_template(discharge_summary, profile):
    templates = {
        "name": {
            "regex": r"Name:  ___",
            "target": f"Name:   ",
            "value": profile["name"],
        },
        "surname_only": {
            "regex": r"Ms. ___",
            "target": f"Ms. ",
            "value": profile["name"],
        },
        "surname_onl2": {
            "regex": r"___ is a",
            "target": f" is a",
            "value": profile["name"],
            "flip": True,
        },
        "surname_only_3": {
            "regex": r"Mr. ___",
            "target": f"Mr. ",
            "value": profile["name"],
        },
        "clinician": {
            "regex": r"Dr. ___",
            "target": f"Dr. ",
            "value": profile["clinician_name"],
        },
        "clinician2": {
            "regex": r"Dr. ___ ___",
            "target": f"Dr. ",
            "value": profile["clinician_name"],
        },
        "clinician3": {
            "regex": r"Dr. \\n___",
            "target": f"Dr.",
            "value": profile["clinician_name"],
        },
        "clinician4": {
            "regex": r"You were seen by ___",
            "target": f"You were seen by",
            "value": profile["clinician_name"],
        },
        "unit_number": {
            "regex": r"Unit No:   ___",
            "target": f"Unit No:",
            "function": generate_random_unit_number,
        },
        "admission_date": {
            "regex": r"Admission Date:  ___",
            "target": f"Admission Date:",
            "value": profile["in_date"],
        },
        "discharge_date": {
            "regex": r"Discharge Date:   ___",
            "target": f"Discharge Date:   ",
            "value": profile["out_date"],
        },
        "date_of_birth": {
            "regex": r"Date of Birth:  ___",
            "target": f"Date of Birth:   ",
            "value": profile["birth_date"],
        },
        "gen_date": {
            "regex": r"on ___",
            "target": f"on ",
            "value": profile["intervention_date"],
        },
        "attending_physician": {
            "regex": r"Attending: ___",
            "target": f"Attending: ",
            "value": profile["clinician_name"],
        },
        "age1": {
            "regex": r"___ year old",
            "target": f" year old",
            "value": profile["age"],
            "flip": True,
        },
        "age2": {
            "regex": r"when pt. was ___",
            "target": f"when pt. was ",
            "value": profile["age"],
            "flip": False,
        },
        "age3": {
            "regex": r"___ man",
            "target": f" man",
            "value": profile["age"],
            "flip": True,
        },
        "age3": {
            "regex": r"___ yo",
            "target": f" yo",
            "value": profile["age"],
            "flip": True,
        },
        "age4": {
            "regex": r"age ___",
            "target": f"age ",
            "value": profile["age"],
        },
        "age4": {
            "regex": r"___ year-old",
            "target": f" year-old",
            "value": profile["age"],
            "flip": True,
        },
        "location": {
            "regex": r"your time at ___",
            "target": f"your time at",
            "value": profile["location"],
        },
        "location2": {
            "regex": r"___ ED",
            "target": f"ED",
            "value": profile["location"],
            "flip": True,
        },
        "location3": {
            "regex": r"taken to ___",
            "target": f"taken to",
            "value": profile["location"],
        },
        "location4": {
            "regex": r" was in ___",
            "target": f" was in",
            "value": profile["location"],
        },
        "location5": {
            "regex": r"presents to ___",
            "target": f"presents to",
            "value": profile["location"],
        },
        "location6": {
            "regex": r"previous in ___",
            "target": f"previous in",
            "value": profile["location"],
        },
        "location7": {
            "regex": r"stay at ___",
            "target": f"stay at",
            "value": profile["location"],
        },
        "location8": {
            "regex": r"___ Care Team",
            "target": f"Care Team",
            "value": profile["location"],
            "flip": True,
        },
        "location9": {
            "regex": r"admitted to ___",
            "target": f"admitted to",
            "value": profile["location"],
        },
        "location10": {
            "regex": r"___ Team",
            "target": f" Team",
            "value": profile["location"],
            "flip": True,
        },
        "location11": {
            "regex": r"you at ___",
            "target": f"you at",
            "value": profile["location"],
        },
        "time": {
            "regex": r"___ weeks",
            "target": f" weeks",
            "function": generate_random_unit_number,
            "flip": True,
        },
    }
    for _, value in templates.items():
        if "function" in value:
            if value.get("flip"):
                discharge_summary = re.sub(
                    pattern=value["regex"],
                    repl=f"{value['function']()} {value['target']}",
                    string=discharge_summary,
                )
            else:
                discharge_summary = re.sub(
                    pattern=value["regex"],
                    repl=f"{value['target']} {value['function']()}",
                    string=discharge_summary,
                )
        else:
            if value.get("flip"):
                discharge_summary = re.sub(
                    pattern=value["regex"],
                    repl=f"{value['value']} {value['target']}",
                    string=discharge_summary,
                )
            else:
                discharge_summary = re.sub(
                    pattern=value["regex"],
                    repl=f"{value['target']} {value['value']}",
                    string=discharge_summary,
                )
    return discharge_summary


def reidentify_ehr_record(target_summary_type, filetype, id, profile, target_input):
    res = fetch_file_names(f"{EXAMPLES_ROOT}{target_input}/{target_summary_type}", filetype, hadm_id=id)
    contents = load_file(res[0])
    data = fill_in_discharge_template(contents, profile)
    data = remove_extra_redactions(data)
    return data


def reidentify_target_summary(target_summary_type, filetype, id, profile, target_input):
    res = fetch_file_names(f"{EXAMPLES_ROOT}{target_input}/{target_summary_type}", "target", hadm_id=id)
    target_summary = load_file(res[0])
    data = fill_in_discharge_template(target_summary, profile)
    data = remove_extra_redactions(data)
    return data


def run_process(hadm_ids, target_input):
    print("Running reidentifier")
    # for both summary tasks
    profiles = []
    for target_summary_type in SUMMARY_TYPES:
        # and for every patient
        for id in hadm_ids:
            print(f"Processing {target_summary_type} for {id}")
            filetype = "discharge-inputs"

            # skip if a reference file is already present
            if reference_file_is_present(
                f"{target_input}/{target_summary_type}{BASELINE_SUMMARY_TASK}", id
            ):
                print(f"Skipping {id} as it has already been processed.")
                continue

            profile = generate_random_profile(id, target_input)
            profiles.append(profile)
            data = reidentify_ehr_record(
                target_summary_type, filetype, id, profile, target_input
            )

            # create folder if it doesn't exist
            if not os.path.exists(
                f"{RE_ID_EXAMPLES_ROOT}/{target_input}/{target_summary_type}"
            ):
                os.makedirs(
                    f"{RE_ID_EXAMPLES_ROOT}/{target_input}/{target_summary_type}"
                )
            with open(
                f"{RE_ID_EXAMPLES_ROOT}/{target_input}/{target_summary_type}/{id}-{filetype}.txt",
                "w",
            ) as f:
                f.write(data)

            baseline_summary = reidentify_target_summary(
                target_summary_type, filetype, id, profile, target_input
            )

            # create folder if it doesn't exist
            if not os.path.exists(
                f"{RE_ID_TARGETS_ROOT}/{target_input}/{target_summary_type}_baseline"
            ):
                os.makedirs(
                    f"{RE_ID_TARGETS_ROOT}/{target_input}/{target_summary_type}_baseline"
                )

            with open(
                f"{RE_ID_TARGETS_ROOT}/{target_input}/{target_summary_type}_baseline/{id}-target.txt",
                "w",
            ) as f:
                f.write(baseline_summary)

        print("Done.")
    with open("data/validation-pseudo-profiles.json", "w") as f:
        json.dump(profiles, f)


if __name__ == "__main__":
    original_discharge_summaries = load_original_discharge_summaries()
    target_admission_ids = extract_hadm_ids(
        original_discharge_summaries=original_discharge_summaries, n=100
    )
    run_process(hadm_ids=target_admission_ids)
