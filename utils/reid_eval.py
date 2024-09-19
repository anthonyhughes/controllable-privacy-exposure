from genericpath import isfile
import json
from os import walk
from constants import (
    PRIVACY_RESULTS_DIR,
    BASELINE_SUMMARY_TASK,
    IN_CONTEXT_SUMMARY_TASK,
    PSEUDO_PROFILES_LOCATION,
    RESULTS_DIR,
    RE_ID_EXAMPLES_ROOT,
    EVAL_MODELS
)
from utils.dataset_utils import open_generated_summary, open_target_summary


def get_all_input_files_for_task(model, task):
    target_file_path = f"{RESULTS_DIR}/{model}/{task}/"
    return next(walk(target_file_path), (None, None, []))[2]


def get_all_profiles():
    with open(PSEUDO_PROFILES_LOCATION) as f:
        return json.load(f)


def run_mapping_to_input_files(target_model, task, all_pseudo_profiles):
    """
    Build a mapping from the pseudo-profiles to the original patient id
    """
    name_to_file_mapping = []
    input_files_for_task = get_all_input_files_for_task(model=target_model, task=task)
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


def run_reidentification_eval(target_model, tasks):
    """
    Run the redentification eval
    """
    all_pseudo_profiles = get_all_profiles()
    print(f'Running reidentification for model: {target_model} and all tasks: {tasks}')
    models = EVAL_MODELS if target_model == "all" else [target_model]
    for model in models:
        for task in tasks:
            print(f'Running reidentification for model: {model} and task: {task}')
            mappings = run_mapping_to_input_files(model, task, all_pseudo_profiles)
            print(f"Found matching files: {len(mappings)}")
            for profile in mappings:
                icl_task = f"{task}{IN_CONTEXT_SUMMARY_TASK}"
                icl_target_summary = open_generated_summary(
                    task=icl_task, hadm_id=profile["hadm_id"], model=model
                )
                for pi_prop, pi_value in profile["profile"].items():
                    if pi_prop in ["age", "gender"]: continue
                    if str(pi_value) in icl_target_summary:
                        print("Re-identification possible")
                        print(pi_value)
                        print(pi_prop)
