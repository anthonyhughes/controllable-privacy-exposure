import os
import re
from constants import PSEUDO_TARGETS_ROOT, SUMMARY_TYPES, EXAMPLE_ADMISSION_IDS
from mimic.mimic_data import load_original_discharge_summaries
from reidentifier.reidentify_utils import (
    fetch_file_names,
    load_file,
    remove_extra_piis,
    remove_extra_redactions,
)
from utils.dataset_utils import extract_hadm_ids


def fill_in_target_summary(discharge_summary):
    age_templates = {
        "age": {
            "regex": r"___ woman",
            "target": f"a patient",
        },
        "age4": {
            "regex": r"___ w ",
            "target": f"Patient",
        },
        "age2": {
            "regex": r"___ yo",
            "target": f"",
        },
        "age3": {
            "regex": r"___ year old",
            "target": f"",
        },
    }
    location_templates = {
        "location": {
            "regex": r"___ Hospital",
            "target": f"a hospital",
        },
        "location2": {
            "regex": r"referred to\n___",
            "target": f"referred to a medical center",
        },
        "location3": {
            "regex": r"PCP ___",
            "target": f"PCP",
        },
        "location4": {
            "regex": r"___ ED",
            "target": f"ED",
        },
        "location5": {
            "regex": r"at ___[.]",
            "target": f"",
        },
        "location6": {
            "regex": r"referred to \n___",
            "target": f"referred to",
        },
        "location7": {
            "regex": r"Your ___ (.+)? Team",
            "target": f"Your Team",
        },
        "location8": {
            "regex": r"Your ___ Team",
            "target": f"Your Team",
        },
        "location9": {
            "regex": r"admitted to ___",
            "target": f"admitted to hospital",
        },
        "location10": {
            "regex": r"___ Medicine",
            "target": f"Medical Team",
        },
        "location111": {
            "regex": r"from ___",
            "target": f"",
        },
        "location12": {
            "regex": r"___ (pain)? clinic",
            "target": f"a clinic",
        },
        "location13": {
            "regex": r"at(\n)?\n___",
            "target": f"at a care facility",
        },
        "location14": {
            "regex": r"through ___",
            "target": f"through a care facility",
        },
    }
    date_templates = {
        "date": {
            "regex": r"[Oo]n ___[,.]",
            "target": f"",
        },
        "date3": {
            "regex": r"on\n ___",
            "target": f"",
        },
        "date4": {
            "regex": r"on \n___",
            "target": f"",
        },
        "date2": {
            "regex": r"in ___",
            "target": f"",
        },
        "date5": {
            "regex": r"on ___",
            "target": f"",
        },
        "date6": {
            "regex": r" narcotics ___",
            "target": f" narcotics",
        },
        "date6": {
            "regex": r"seen ___",
            "target": f"seen",
        },
        "date6": {
            "regex": r"completed ___",
            "target": f"completed",
        },
    }
    clinician_templates = {
        "clinician": {
            "regex": r"Dr. ___",
            "target": f"a doctor",
        },
        "clinician2": {
            "regex": r"Dr.\n___",
            "target": f"a doctor",
        },
        "clinician3": {
            "regex": r"Dr. \n___",
            "target": f"a doctor",
        },
        "clinician4": {
            "regex": r"seen by ___",
            "target": f"seen by a doctor",
        },
    }
    patient_templates = {
        "patient": {
            "regex": r"Ms[.]? ___",
            "target": f"The patient",
        },
        "patient5": {
            "regex": r"Mr[.]? ___",
            "target": f"The patient",
        },
        "patient2": {
            "regex": r"___ was contacted",
            "target": f"the patient was contacted",
        },
        "patient12": {
            "regex": r"___ is a ___ with",
            "target": f"Patient has",
        },
        "patient3": {
            "regex": r"is a ___ with",
            "target": f"has",
        },
        "patient10": {
            "regex": r"___ with a",
            "target": f"Patient with a",
        },
        "patient11": {
            "regex": r"^___ with ",
            "target": f"Patient with ",
        },
        "patient4": {
            "regex": r"^___ with",
            "target": f"Patient has",
        },
        "patient6": {
            "regex": r"___ speaking",
            "target": f"",
        },
        "patient7": {
            "regex": r"___ patient",
            "target": f"A patient",
        },
        "patient8": {
            "regex": r"for ___",
            "target": f"for the patient",
        },
        "patient9": {
            "regex": r"Dear ___[,]",
            "target": f"Dear patient",
        },
    }
    other_templates = {
        "other": {
            "regex": r"and ___",
            "target": f"",
        },
        "other2": {
            "regex": r"as ___",
            "target": f"",
        },
        "other3": {
            "regex": r"called ___",
            "target": f"",
        },
    }
    templates = {
        **age_templates,
        **location_templates,
        **date_templates,
        **clinician_templates,
        **patient_templates,
        **other_templates,
    }
    for _, value in templates.items():
        discharge_summary = re.sub(
            pattern=value["regex"],
            repl=f"{value['target']}",
            string=discharge_summary,
        )
    return discharge_summary


def run_pseudonmizer_process(task, hadm_ids):
    """
    Run the re-identification process
    """
    print("Running pseudonimizer")
    for id in hadm_ids:
        # print(res)
        print(f"Processing {task} for {id}")
        res = fetch_file_names(f"data/examples/{task}", "target", hadm_id=id)
        contents = load_file(res[0])
        data = fill_in_target_summary(contents)
        data = remove_extra_redactions(data)
        data =  remove_extra_piis(data)
        if not os.path.exists(f"{PSEUDO_TARGETS_ROOT}/{task}"):
            os.makedirs(f"{PSEUDO_TARGETS_ROOT}/{task}")
        with open(
            f"{PSEUDO_TARGETS_ROOT}/{task}/{id}-target.txt",
            "w",
        ) as f:
            f.write(data)
    print("Done.")


def run_all_pseudonmizer_processes(hadm_ids):
    for target_summary_type in SUMMARY_TYPES:
        run_pseudonmizer_process(target_summary_type, hadm_ids)


if __name__ == "__main__":
    original_discharge_summaries = load_original_discharge_summaries()
    hadm_ids = extract_hadm_ids(original_discharge_summaries)
    run_all_pseudonmizer_processes(hadm_ids)
