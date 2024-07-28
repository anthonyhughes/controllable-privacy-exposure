from mimic.mimic_data import run
from reidentifier.re_identifier import run_process
from pseudonymizer.pseudonymizer import run_all_pseudonmizer_processes

if __name__ == "__main__":
    print("Starting data pipeline")
    run()
    run_process()
    run_all_pseudonmizer_processes()
