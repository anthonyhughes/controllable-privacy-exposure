import re
import pandas as pd

from utils.dataset_utils import extract_hadm_ids

pd.set_option("display.max_columns", None)
from constants import (
    TRAIN_DISCHARGE_ME,
    EXAMPLES_ROOT,
    TRAIN_DISCHARGE_ME,
    SUMMARY_TYPES,
)


def load_target_summaries():
    return pd.read_csv(TRAIN_DISCHARGE_ME + "discharge_target.csv")


def load_radiology_summaries():
    return pd.read_csv(TRAIN_DISCHARGE_ME + "radiology.csv")


def load_original_discharge_summaries():
    return pd.read_csv(TRAIN_DISCHARGE_ME + "discharge.csv")


def save_example_to_file(filename, content_to_save):
    print("Saving training data for one patient:")
    with open(EXAMPLES_ROOT + filename, "w") as f:
        f.write(content_to_save)


def preprocessing_of_discharge_summaries(df_target, target_summary):
    """
    Remove a specific target summary from the discharge summaries.
    Necessary to not introduce bias into the system.
    """
    # r"Brief Hospital Course:\s*\n{0,2}(.*?)(?=\n\s*\n{0,2}\s*[A-Z_]+[^\n:]+:\n)"
    # r"Discharge Instructions:\n(.*?)Followup Instruction"

    if target_summary == "discharge_instructions":
        df_target["text"] = df_target["text"].apply(
            lambda x: re.sub(
                r"Discharge Instructions:\n(.*?)Followup Instruction",
                "",
                x,
                flags=re.DOTALL,
            )
        )
    elif target_summary == "brief_hospital_course":
        df_target["text"] = df_target["text"].apply(
            lambda x: re.sub(
                r"Brief Hospital Course:\s*\n{0,2}(.*?)(?=\n\s*\n{0,2}\s*[A-Z_]+[^\n:]+:\n)",
                "",
                x,
                flags=re.DOTALL,
            )
        )
    return df_target


def extract_sample_for_admission(
    hadm_id,
    original_discharge_summaries,
    radiology_summaries,
    target_summaries,
    target_summary_type,
):
    print("Example admission ID:", hadm_id)
    example_target_summaries = target_summaries[target_summaries["hadm_id"] == hadm_id]
    target_summaries_out = ""

    if target_summary_type == "discharge_instructions":
        # target_summaries_out += "\n\n##Discharge Target Summary###\n\n"
        discharge_target = example_target_summaries["discharge_instructions"].values[0]
        target_summaries_out += discharge_target

    if target_summary_type == "brief_hospital_course":
        # target_summaries_out += "\n\n##Brief Hospital Course Summary###\n\n"
        bhc_target = example_target_summaries["brief_hospital_course"].values[0]
        target_summaries_out += bhc_target

    save_example_to_file(
        f"{target_summary_type}/{hadm_id}-target.txt", target_summaries_out
    )

    training_summaries = ""
    # training_summaries += "\n\n##Original Discharge##\n\n"
    print("Loading original discharge summaries:")
    example_original_discharge_summaries = original_discharge_summaries[
        original_discharge_summaries["hadm_id"] == hadm_id
    ]
    original_discharge = example_original_discharge_summaries["text"].values[0]
    training_summaries += original_discharge
    save_example_to_file(
        f"{target_summary_type}/{hadm_id}-discharge-inputs.txt", training_summaries
    )

    print("Loading radiology summaries:")
    example_radiology_summaries = radiology_summaries[
        radiology_summaries["hadm_id"] == hadm_id
    ]

    radiology_summaries_out = ""
    for i, row in example_radiology_summaries.iterrows():
        # radiology_summaries_out += f"\n\n#Radiology Summary {i}#\n\n"
        r_summary = row["text"]
        radiology_summaries_out += r_summary
    save_example_to_file(
        f"{target_summary_type}/{hadm_id}-radiology-inputs.txt", radiology_summaries_out
    )

    print("Done.")


def run():
    print("Starting extraction of mimic data")
    for target_summary_type in SUMMARY_TYPES:
        original_discharge_summaries = load_original_discharge_summaries()

        original_discharge_summaries = preprocessing_of_discharge_summaries(
            original_discharge_summaries, target_summary_type
        )
        radiology_summaries = load_radiology_summaries()

        print("Loading target summaries:")
        target_summaries = load_target_summaries()

        # extract the first 100 admission ids as a list
        target_admission_ids = extract_hadm_ids(
            original_discharge_summaries=original_discharge_summaries, n=10000
        )

        for example_admission_id in target_admission_ids:
            extract_sample_for_admission(
                example_admission_id,
                original_discharge_summaries,
                radiology_summaries,
                target_summaries,
                target_summary_type,
            )
    print("Done.")
    return target_admission_ids


if __name__ == "__main__":
    run()
