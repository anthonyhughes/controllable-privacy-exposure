import os
from constants import ICL_EXAMPLES_ROOT, PSEUDO_TARGETS_ROOT, RE_ID_EXAMPLES_ROOT, RE_ID_TARGETS_ROOT, DEIDENTIFICATION_DICT
from pseudonymizer.pseudo_utils import deidentify_text
from utils.dataset_utils import open_legal_data, open_validation_legal_data


def run_ls_pseudonmizer_processes(deidentifier, task="legal_court", target_input_set="valid"):
    """
    Psuedonymize legal docs
    """
    print("Start psuedo process for legal court summaries")
    legal_data = open_legal_data()
    icl_examples = [
        "legalsum81",
        "legalsum82",
        "legalsum83",
        "legalsum84",
        "legalsum85",
    ]
    for i, key in enumerate(legal_data.keys()):
        print(f"Started doc {i+1} of {len(legal_data.keys())}")
        doc = legal_data[key]
        target_summary = doc["target"]

        scrubbed_text = deidentifier.deidentify(target_summary)

        replaced_scrubbed_text = deidentify_text(
            text=scrubbed_text.text, deidentification_dict=DEIDENTIFICATION_DICT
        )

        if key not in icl_examples:
            # Prebuild folder for main summaries (non-deidentified)
            if not os.path.exists(f"{RE_ID_TARGETS_ROOT}/{target_input_set}/{task}"):
                os.makedirs(f"{RE_ID_TARGETS_ROOT}/{target_input_set}/{task}")

            # Store document per legal doc
            with open(
                f"{RE_ID_TARGETS_ROOT}/{target_input_set}/{task}/{key}-target.txt",
                "w",
            ) as f:
                f.write(target_summary)

            # Prebuild folder for main inputs
            main_legal_doc = doc["document"]
            if not os.path.exists(f"{RE_ID_EXAMPLES_ROOT}/{target_input_set}/{task}"):
                os.makedirs(f"{RE_ID_EXAMPLES_ROOT}/{target_input_set}/{task}")

            # Store document per legal doc
            with open(
                f"{RE_ID_EXAMPLES_ROOT}/{target_input_set}/{task}/{key}-discharge-inputs.txt",
                "w",
            ) as f:
                f.write(main_legal_doc)

            # Prebuild folder for psudeo summaries
            if not os.path.exists(f"{PSEUDO_TARGETS_ROOT}/{target_input_set}/{task}"):
                os.makedirs(f"{PSEUDO_TARGETS_ROOT}/{target_input_set}/{task}")

            # Store document per legal summary
            with open(
                f"{PSEUDO_TARGETS_ROOT}{target_input_set}/{task}/{key}-target.txt",
                "w",
            ) as f:
                f.write(replaced_scrubbed_text)
        else:
            # Prebuild folder for psudeo summaries
            if not os.path.exists(f"{ICL_EXAMPLES_ROOT}/{task}"):
                os.makedirs(f"{ICL_EXAMPLES_ROOT}/{task}")

            # Store document per legal summary
            with open(
                f"{ICL_EXAMPLES_ROOT}/{task}/{key}-target.txt",
                "w",
            ) as f:
                f.write(replaced_scrubbed_text)
