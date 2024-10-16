import os
from constants import (
    ICL_EXAMPLES_ROOT,
    PSEUDO_TARGETS_ROOT,
    RE_ID_EXAMPLES_ROOT,
    RE_ID_TARGETS_ROOT,
    DEIDENTIFICATION_DICT,
    SANITIZED_INPUTS_ROOT,
)
from pseudonymizer.pseudo_utils import deidentify_text, sanitize_text
from utils.dataset_utils import open_cnn_data, open_cnn_train_data


def run_cnn_pseudonmizer_processes(deidentifier, task="cnn", target_input_set="valid"):
    """
    Psuedonymize legal docs
    """
    print("Start psuedo process for legal court summaries")
    cnn_data = (
        open_cnn_train_data()
        if (target_input_set == "valid" or target_input_set == "train")
        else open_cnn_data()
    )
    ids = list(cnn_data.keys())
    icl_example_ids = ids[-5:]
    ids = ids[0:5000]
    for i, key in enumerate(ids):
        print(f"Started doc {i+1} of {len(cnn_data.keys())}")
        doc = cnn_data[key]
        input_document = doc["document"]
        target_summary = doc["target"]

        scrubbed_input_text = deidentifier.deidentify(input_document)
        scrubbed_text = deidentifier.deidentify(target_summary)

        sanitized_input_doc = sanitize_text(
            text=scrubbed_input_text.text, deidentification_dict=DEIDENTIFICATION_DICT
        )
        replaced_scrubbed_text = deidentify_text(
            text=scrubbed_text.text, deidentification_dict=DEIDENTIFICATION_DICT
        )

        if key not in icl_example_ids:
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
                f"{PSEUDO_TARGETS_ROOT}/{target_input_set}/{task}/{key}-target.txt",
                "w",
            ) as f:
                f.write(replaced_scrubbed_text)

            # Prebuild folder for sanitized inputs
            if not os.path.exists(f"{SANITIZED_INPUTS_ROOT}/{target_input_set}/{task}"):
                os.makedirs(f"{SANITIZED_INPUTS_ROOT}/{target_input_set}/{task}")

            # Store document per input
            with open(
                f"{SANITIZED_INPUTS_ROOT}/{target_input_set}/{task}/{key}.txt",
                "w",
            ) as f:
                f.write(sanitized_input_doc)
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
