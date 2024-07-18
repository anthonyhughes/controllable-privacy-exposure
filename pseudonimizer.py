from glob import glob
import os
import re


def fetch_file_names(path, type, hadm_id="*"):
    files = list(glob(os.path.join(f"{path}", f"{hadm_id}-{type}.txt")))
    return files


def load_file(file):
    with open(file, "r") as f:
        return f.read()


def fill_in_discharge_template(discharge_summary):
    templates = [
        r"Name:  ___",
        # r"Unit No:   ___",
        # r"Admission Date:  ___",
        # r"Discharge Date:   ___",
        # r"Date of Birth:  ___"
    ]
    result = re.sub(
        pattern=templates[0],
        repl="Name:  John Doe",
        string=discharge_summary,
    )
    return result


def run_process():
    print("Running pseudonimizer")
    print("Load files for pseudonimization")
    # load all input files from data/examples/brief_hospital_course and data/examples/discharge_instructions
    res = fetch_file_names(
        "data/examples/brief_hospital_course", "discharge-inputs", "22907047"
    )
    # print(res)
    contents = load_file(res[0])
    print(contents)
    data = fill_in_discharge_template(contents)[0:100]
    print(data)

    # res = fetch_file_names(
    #     "data/examples/brief_hospital_course", "radiology-inputs", "22907047"
    # )
    # print(res)
    # res = fetch_file_names("data/examples/brief_hospital_course", "target", "22907047")
    # print(res)
    # pseudonimize the text in the files
    # save the pseudonimized text in the same files
    print("Done.")


if __name__ == "__main__":
    run_process()
