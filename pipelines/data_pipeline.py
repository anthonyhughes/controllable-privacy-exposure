from mimic.mimic_data import run
from reidentifier.re_identifier import run_process
from pseudonymizer.pseudonymizer import run_all_pseudonmizer_processes

if __name__ == "__main__":
    print("Starting data pipeline")
    # Mimic extraction
    hadm_ids = run()
    # Add pseudo data to mimic data
    run_process(hadm_ids)
    # Pseudonymizer of the summaries
    run_all_pseudonmizer_processes(hadm_ids)
    # run_packaging_for_colab()
    print("Done.")