# This pipeline requires manually install of unsloth
# pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"
import argparse
from unsloth import FastLanguageModel
from constants import (
    BASELINE_SUMMARY_TASK,
    EVAL_MODELS,
    IN_CONTEXT_SUMMARY_TASK,
    SANI_SUMM_SUMMARY_TASK,
    SUMMARY_TYPES,
    RE_ID_EXAMPLES_ROOT,
    RESULTS_DIR,
)
from utils.dataset_utils import (
    extract_hadm_ids_from_dir,
    fetch_example,
    open_cnn_data,
    open_legal_data,
    read_file,
    write_to_file,
    create_missing_output_folders,
)
from utils.prompt_variations import instruction_prompt, variations, sanitize_prompt
import os


def get_model(target_model):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/" + target_model,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def run_inference(instruction, input, model, tokenizer):
    inputs = tokenizer(
        [
            instruction_prompt.format(
                instruction,
                input,
                "",  # output - leave this blank for generation!
            )
        ],
        return_tensors="pt",
    ).to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=2048, use_cache=True)
    result = tokenizer.batch_decode(outputs)
    trimmed_res = result[0].split("### Response:")[1].strip()
    if "### Explanation" in trimmed_res:
        trimmed_res = trimmed_res.split("### Explanation")[0].strip()
    return trimmed_res


def is_excluded_id(id):
    return id in [
        "24186608",
        "23307632",
        "23709960",
        "21888122",
        "28709120",
        "27206728",
        "24695226",
        "21540678",
        "24862430"
    ]


def run_all_inference(target_model, hadm_ids, summary_type):
    # for each summary type
    model, tokenizer = get_model(target_model=target_model)
    for prompt_prefix_for_task in variations:
        v_name = prompt_prefix_for_task["name"]
        print(
            f"Running inference for summary type : {summary_type} and variation: {v_name} on model: {target_model}"
        )
        # for both a baseline summary and an instructed to be pseudo summary
        instruction_main = prompt_prefix_for_task[summary_type]
        instruction_baseline = prompt_prefix_for_task[
            f"{summary_type}{BASELINE_SUMMARY_TASK}"
        ]
        instruction_icl = prompt_prefix_for_task[
            f"{summary_type}{IN_CONTEXT_SUMMARY_TASK}"
        ]
        instruction_sani_summ = prompt_prefix_for_task[
            f"{summary_type}{SANI_SUMM_SUMMARY_TASK}"
        ]
        print(f"Running inference for {len(hadm_ids)} identities")
        for i, id in enumerate(hadm_ids):
            if is_excluded_id(id):
                print(f'Skipping {id}')
                continue

            print(f"Running inference for file {i+1}/{len(hadm_ids)}")
            print(
                f"Running inference for model {target_model}, task {summary_type} and variation {v_name}"
            )
            # Set ins
            input_filename = (
                f"{RE_ID_EXAMPLES_ROOT}/{summary_type}/{id}-discharge-inputs.txt"
            )
            # Set outs
            output_filename = f"{RESULTS_DIR}/{target_model}/{v_name}/{summary_type}/{id}-discharge-inputs.txt"
            baseline_output_file = f"{RESULTS_DIR}/{target_model}/{v_name}/{summary_type}{BASELINE_SUMMARY_TASK}/{id}-discharge-inputs.txt"
            icl_output_file = f"{RESULTS_DIR}/{target_model}/{v_name}/{summary_type}{IN_CONTEXT_SUMMARY_TASK}/{id}-discharge-inputs.txt"
            sani_summ_output_file = f"{RESULTS_DIR}/{target_model}/{v_name}/{summary_type}{SANI_SUMM_SUMMARY_TASK}/{id}-discharge-inputs.txt"

            # extra file for the sanitized document
            sani_summ_sanitized_output_file = f"{RESULTS_DIR}/{target_model}/{v_name}/{summary_type}_sanitized/{id}-discharge-inputs.txt"

            # Creating results dir
            create_missing_output_folders(target_model, v_name, summary_type)

            print(f"Starting inference for {id}")
            file_input = read_file(input_filename)

            # if the output file already exists, skip it (inference already completed)
            if (
                file_input is not None
                and os.path.exists(output_filename) is False
            ):
                print("Privacy Baseline")
                # get a pseduo summary
                pseudo_trimmed_res = run_inference(
                    instruction_main, file_input, model, tokenizer
                )
                # write pseudonimysed output
                write_to_file(output_filename, pseudo_trimmed_res)

            if file_input is not None and os.path.exists(baseline_output_file) is False:
                print("Instruction Baseline")
                # get a baseline (non-pseduo summary)
                baseline_trimmed_res = run_inference(
                    instruction_baseline, file_input, model, tokenizer
                )
                # write baseline output
                write_to_file(baseline_output_file, baseline_trimmed_res)

            if (
                file_input is not None
                and os.path.exists(icl_output_file) is False
            ):
                print("ICL")
                # get a baseline (non-pseduo summary)
                # add an in-context example
                instruction_icl = instruction_icl.replace(
                    "[incontext_examples]", f"{fetch_example(summary_type)}"
                )
                icl_trimmed_res = run_inference(
                    instruction_icl, file_input, model, tokenizer
                )
                # write baseline output
                write_to_file(icl_output_file, icl_trimmed_res)

            if (
                file_input is not None
                and os.path.exists(sani_summ_output_file) is False
            ):
                print("SaniSumm - Sani")
                if not os.path.exists(
                    f"{RESULTS_DIR}/{target_model}/variation_1/{summary_type}_sanitized/{id}-discharge-inputs.txt"
                ):
                    print("Using LLM to sanitize")
                    sanitized_document = run_inference(
                        sanitize_prompt, file_input, model, tokenizer
                    )
                    write_to_file(sani_summ_sanitized_output_file, sanitized_document)
                else:
                    print("Reusing prior sanitized document")
                    sanitized_document = read_file(
                        f"{RESULTS_DIR}/{target_model}/variation_1/{summary_type}_sanitized/{id}-discharge-inputs.txt"
                    )
                print("SaniSumm - Summ")
                sani_summ_trimmed_res = run_inference(
                    instruction_sani_summ, sanitized_document, model, tokenizer
                )
                write_to_file(sani_summ_output_file, sani_summ_trimmed_res)
            print(f"ID complete {id}")


def main():
    print("Starting inference")
    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--task",
        help="Choose a task for inference",
        default=SUMMARY_TYPES[2],
        choices=SUMMARY_TYPES,
    )
    parser.add_argument(
        "-m",
        "--model",
        help="Choose a target model for inference",
        default=EVAL_MODELS[4],
        choices=EVAL_MODELS,
    )
    args = parser.parse_args()

    hadm_ids = []
    if args.task == "legal_court":
        legal_data = open_legal_data()
        hadm_ids = sorted(list(legal_data.keys()))
        hadm_ids = hadm_ids[0:-5]
    elif args.task == "cnn":
        cnn_data = open_cnn_data()
        hadm_ids = sorted(list(cnn_data.keys()))
        hadm_ids = hadm_ids[0:-5]
    else:
        hadm_ids = extract_hadm_ids_from_dir(
            "llama-3-8b-Instruct-bnb-4bit", "brief_hospital_course", "variation_1"
        )
    run_all_inference(
        target_model=args.model, hadm_ids=hadm_ids, summary_type=args.task
    )
    print("Inference complete")


if __name__ == "__main__":
    main()
