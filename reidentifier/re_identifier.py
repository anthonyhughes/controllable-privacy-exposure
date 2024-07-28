import os
import re
from constants import RE_ID_EXAMPLES_ROOT, SUMMARY_TYPES, EXAMPLE_ADMISSION_IDS
from reidentifier.reidentify_utils import (
    fetch_file_names,
    load_file,
    generate_random_name,
    generate_random_unit_number,
    generate_random_adult_age,
    generate_random_date,
    generate_random_location,
    remove_extra_redactions,
)


def fill_in_discharge_template(discharge_summary):
    templates = {
        "name": {
            "regex": r"Name:  ___",
            "target": f"Name:   ",
            "function": generate_random_name,
        },
        "surname_only": {
            "regex": r"Ms. ___",
            "target": f"Ms. ",
            "function": generate_random_name,
        },
        "surname_onl2": {
            "regex": r"___ is a",
            "target": f" is a",
            "function": generate_random_name,
            "flip": True,
        },
        "surname_only_3": {
            "regex": r"Mr. ___",
            "target": f"Mr. ",
            "function": generate_random_name,
        },
        "clinician": {
            "regex": r"Dr. ___",
            "target": f"Dr. ",
            "function": generate_random_name,
        },
        "clinician2": {
            "regex": r"Dr. ___ ___",
            "target": f"Dr. ",
            "function": generate_random_name,
        },
        "clinician3": {
            "regex": r"Dr. \\n___",
            "target": f"Dr.",
            "function": generate_random_name,
        },
        "clinician4": {
            "regex": r"You were seen by ___",
            "target": f"You were seen by",
            "function": generate_random_name,
        },
        "unit_number": {
            "regex": r"Unit No:   ___",
            "target": f"Unit No:",
            "function": generate_random_unit_number,
        },
        "admission_date": {
            "regex": r"Admission Date:  ___",
            "target": f"Admission Date:",
            "function": generate_random_date,
        },
        "discharge_date": {
            "regex": r"Discharge Date:   ___",
            "target": f"Discharge Date:   ",
            "function": generate_random_date,
        },
        "date_of_birth": {
            "regex": r"Date of Birth:  ___",
            "target": f"Date of Birth:   ",
            "function": generate_random_date,
        },
        "gen_date": {
            "regex": r"on ___",
            "target": f"on ",
            "function": generate_random_date,
        },
        "attending_physician": {
            "regex": r"Attending: ___",
            "target": f"Attending: ",
            "function": generate_random_name,
        },
        "age1": {
            "regex": r"___ year old",
            "target": f" year old",
            "function": generate_random_adult_age,
            "flip": True,
        },
        "age2": {
            "regex": r"when pt. was ___",
            "target": f"when pt. was ",
            "function": generate_random_adult_age,
            "flip": False,
        },
        "age3": {
            "regex": r"___ man",
            "target": f" man",
            "function": generate_random_adult_age,
            "flip": True,
        },
        "age3": {
            "regex": r"___ yo",
            "target": f" yo",
            "function": generate_random_adult_age,
            "flip": True,
        },
        "age4": {
            "regex": r"age ___",
            "target": f"age ",
            "function": generate_random_adult_age,
        },
        "age4": {
            "regex": r"___ year-old",
            "target": f" year-old",
            "function": generate_random_adult_age,
            "flip": True,
        },
        "location": {
            "regex": r"your time at ___",
            "target": f"your time at",
            "function": generate_random_location,
        },
        "location2": {
            "regex": r"___ ED",
            "target": f"ED",
            "function": generate_random_location,
            "flip": True,
        },
        "location3": {
            "regex": r"taken to ___",
            "target": f"taken to",
            "function": generate_random_location,
        },
        "location4": {
            "regex": r" was in ___",
            "target": f" was in",
            "function": generate_random_location,
        },
        "location5": {
            "regex": r"presents to ___",
            "target": f"presents to",
            "function": generate_random_location,
        },
        "location6": {
            "regex": r"previous in ___",
            "target": f"previous in",
            "function": generate_random_location,
        },
        "location7": {
            "regex": r"stay at ___",
            "target": f"stay at",
            "function": generate_random_location,
        },
        "location8": {
            "regex": r"___ Care Team",
            "target": f"Care Team",
            "function": generate_random_location,
            "flip": True,
        },
        "location9": {
            "regex": r"admitted to ___",
            "target": f"admitted to",
            "function": generate_random_location,
        },
        "location10": {
            "regex": r"___ Team",
            "target": f" Team",
            "function": generate_random_location,
            "flip": True,
        },
        "location11": {
            "regex": r"you at ___",
            "target": f"you at",
            "function": generate_random_location,
        },
        "time": {
            "regex": r"___ weeks",
            "target": f" weeks",
            "function": generate_random_unit_number,
            "flip": True,
        },
    }
    for _, value in templates.items():
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
    return discharge_summary


def run_process():
    print("Running reidentifier")
    # for both summary tasks
    for target_summary_type in SUMMARY_TYPES:
        # and for every patient
        for id in EXAMPLE_ADMISSION_IDS:        
            print(f"Processing {target_summary_type} for {id}")
            filetype = "discharge-inputs"
            res = fetch_file_names(
                f"data/examples/{target_summary_type}", filetype, hadm_id=id
            )
            contents = load_file(res[0])
            data = fill_in_discharge_template(contents)
            data = remove_extra_redactions(data)
            if not os.path.exists(f"{RE_ID_EXAMPLES_ROOT}/{target_summary_type}"):
                os.makedirs(f"{RE_ID_EXAMPLES_ROOT}/{target_summary_type}")
            with open(
                f"{RE_ID_EXAMPLES_ROOT}/{target_summary_type}/{id}-{filetype}.txt", "w"
            ) as f:
                f.write(data)
        print("Done.")


if __name__ == "__main__":
    run_process()
