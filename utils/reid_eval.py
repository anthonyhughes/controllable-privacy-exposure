import json
from os import walk
import os
from constants import (
    IN_CONTEXT_SUMMARY_TASK,
    PSEUDO_PROFILES_LOCATION,
    RESULTS_DIR,
    RE_ID_EXAMPLES_ROOT,
    EVAL_MODELS,
    DATA_ROOT,
    SANI_SUMM_SUMMARY_TASK,
)
from utils.dataset_utils import open_generated_summary, write_to_file
import time


def get_all_input_files_for_task(model, task, variation):
    target_file_path = f"{RESULTS_DIR}/{model}/{variation}/{task}/"
    return next(walk(target_file_path), (None, None, []))[2]


def get_all_profiles():
    with open(PSEUDO_PROFILES_LOCATION) as f:
        return json.load(f)


def run_mapping_to_input_files(target_model, task, all_pseudo_profiles, variation):
    """
    Build a mapping from the pseudo-profiles to the original patient id
    """
    print("Gen the mapping file")
    name_to_file_mapping = []
    input_files_for_task = get_all_input_files_for_task(
        model=target_model, task=task, variation=variation
    )
    for file in input_files_for_task:
        with open(f"{RE_ID_EXAMPLES_ROOT}/{task}/{file}") as f:
            file_lines = f.readlines()
            target_lines = "\n".join(file_lines[0:10])
            for profile in all_pseudo_profiles:
                name = profile["name"]
                if name in target_lines:
                    name_to_file_mapping.append(
                        {
                            "name": name,
                            "hadm_id": file.split("-")[0],
                            "file": file,
                            "task": task,
                            "profile": profile,
                        }
                    )
    return name_to_file_mapping


def identify_an_individual(task, profile, model, variation):
    target_summary = open_generated_summary(
        task=task, hadm_id=profile["hadm_id"], model=model, variation=variation
    )
    for pi_prop, pi_value in profile["profile"].items():
        if pi_prop in ["age", "gender"]:
            continue
        if str(pi_value) in target_summary:
            print("Re-identification possible")
            print(pi_value)
            print(pi_prop)
            return True
    return False


def update_task_results(identified_persons, model, task):
    found_profs = [
        person
        for person in identified_persons[model][task]["results"]
        if person is True
    ]
    identified_persons[model][task]["identifiable"] = len(found_profs)
    identified_persons[model][task]["documents"] = len(
        identified_persons[model][task]["results"]
    )
    try:
        identified_persons[model][task]["ratio"] = len(found_profs) / len(
            identified_persons[model][task]["results"]
        )
    except ZeroDivisionError:
        identified_persons[model][task]["ratio"] = 0
    return identified_persons


def run_reidentification_eval(target_model, tasks, variation, sub_tasks=[]):
    """
    Run the redentification eval
    """
    start_time = time.time()
    all_pseudo_profiles = get_all_profiles()
    models = EVAL_MODELS if target_model == "all" else [target_model]
    identified_persons = {}
    for model in models:
        identified_persons[model] = {}
        for task in tasks:
            identified_persons[model][task] = {}
            identified_persons[model][task]["results"] = []

            icl_task = f"{task}{IN_CONTEXT_SUMMARY_TASK}"
            identified_persons[model][icl_task] = {}
            identified_persons[model][icl_task]["results"] = []

            ss_task = f"{task}{SANI_SUMM_SUMMARY_TASK}"
            identified_persons[model][ss_task] = {}
            identified_persons[model][ss_task]["results"] = []

            mappings = run_mapping_to_input_files(
                model, task, all_pseudo_profiles, variation
            )
            print(f"Found matching files: {len(mappings)}")
            for profile in mappings:
                print(
                    f"Running reidentification for model: {model}, variation: {variation} and task: {task}"
                )
                try:
                    result = identify_an_individual(
                        f"{task}", profile, model, variation
                    )
                    identified_persons[model][task]["results"].append(result)
                except FileNotFoundError:
                    print("File not found")
                    identified_persons[model][task]["results"].append(False)
                    continue

                print(
                    f"Running reidentification for model: {model}, variation: {variation} and task: {icl_task}"
                )
                try:
                    result = identify_an_individual(icl_task, profile, model, variation)
                    identified_persons[model][icl_task]["results"].append(result)
                except FileNotFoundError:
                    print("File not found")
                    identified_persons[model][icl_task]["results"].append(False)
                    continue

                print(
                    f"Running reidentification for model: {model}, variation: {variation} and task: {ss_task}"
                )
                try:
                    result = identify_an_individual(ss_task, profile, model, variation)
                    identified_persons[model][ss_task]["results"].append(result)
                except FileNotFoundError:
                    print("File not found")
                    continue

    print("Taking count")
    for model in models:
        for task in sub_tasks:
            identified_persons = update_task_results(identified_persons, model, task)
            identified_persons = update_task_results(
                identified_persons, model, icl_task
            )
            identified_persons = update_task_results(identified_persons, model, ss_task)

    print("Writing results")
    if not os.path.exists(f"{DATA_ROOT}/re_identification_results/"):
        os.makedirs(f"{DATA_ROOT}/re_identification_results/")

    write_to_file(
        f"{DATA_ROOT}/re_identification_results/results.json",
        json.dumps(identified_persons, indent=4),
    )
    endtime = time.time() - start_time
    print(f"Time taken: {endtime}")
