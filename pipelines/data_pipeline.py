from mimic.mimic_data import run
from reidentifier.re_identifier import run_process
from pseudonymizer.pseudonymizer import run_all_pseudonmizer_processes
from utils.dataset_utils import extract_hadm_ids_from_dir

if __name__ == "__main__":
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