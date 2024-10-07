import argparse
from mimic.mimic_data import run
from reidentifier.re_identifier import run_process
from utils.dataset_utils import add_training_data_to_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--flag", help="Choose a dataset to produce evaluation", default="test", choices=["test", "training"]
    )
    args = parser.parse_args()
    if args.flag == "test":
        print("Starting data pipeline")
        # Mimic extraction
        target_input = "valid"
        hadm_ids = run(with_extraction=False, target_input_set=target_input)
        # hadm_ids = extract_hadm_ids_from_dir("gpt-4o-mini", "brief_hospital_course")
        # Add pseudo-identification data to mimic data
        sorted_ids = sorted(hadm_ids)
        run_process(hadm_ids, target_input=target_input)
        # Pseudonymizer of the summaries
        # run_all_pseudonmizer_processes(hadm_ids)
        # run_packaging_for_colab()
        print("Done.")
    elif args.flag == "training":
        print("Starting training data pipeline")
        add_training_data_to_csv()

        