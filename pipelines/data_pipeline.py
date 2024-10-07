import argparse
from mimic.mimic_data import run
from reidentifier.re_identifier import run_process
from pseudonymizer.pseudonymize import run_all_pseudonmizer_processes
from utils.dataset_utils import extract_hadm_ids_from_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--flage", help="Choose a dataset to produce evaluation", default=["test", "training"]
    )
    args = parser.parse_args()
    if args.flag == "test":
        print("Starting data pipeline")
        # Mimic extraction
        # hadm_ids = run(with_extraction=False)
        hadm_ids = extract_hadm_ids_from_dir("gpt-4o-mini", "brief_hospital_course")
        # Add pseudo-identification data to mimic data
        run_process(hadm_ids)
        # Pseudonymizer of the summaries
        run_all_pseudonmizer_processes(hadm_ids)
        # run_packaging_for_colab()
        print("Done.")
    elif args.flag == "training":
        print("Starting training data pipeline")
        hadm_ids = extract_hadm_ids_from_dir("gpt-4o-mini", "brief_hospital_course")

        