# This pipeline requires manually install of unsloth
# pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"
import argparse
from unsloth import FastLanguageModel
from constants import EVAL_MODELS, SUMMARY_TYPES, RE_ID_EXAMPLES_ROOT, RESULTS_DIR
from utils.dataset_utils import extract_hadm_ids_from_dir, fetch_example, open_cnn_data, open_legal_data, read_file, write_to_file
from utils.prompt_variations import (
    instruction_prompt,
    variations,
)
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
    return trimmed_res


def run_all_inference(target_model, hadm_ids, summary_type):
    # for each summary type
    model, tokenizer = get_model(target_model=target_model)
    for prompt_prefix_for_task in variations:
        v_name = prompt_prefix_for_task["name"]
        print(f'Running inference for summary type : {summary_type} and variation: {v_name} on model: {target_model}')
        # for both a baseline summary and an instructed to be pseudo summary
        instruction_main = prompt_prefix_for_task[summary_type]
        instruction_baseline = prompt_prefix_for_task[f"{summary_type}_baseline"]
        instruction_icl = prompt_prefix_for_task[f"{summary_type}_in_context"]
        print(f"Running inference for {len(hadm_ids)} identities")
        for i, id in enumerate(hadm_ids):
            print(f"Running inference for file {i+1}/{len(hadm_ids)}")
            print(f"Running inference for mode {target_model} and variation {v_name}")
            # Set ins
            input_filename = f"{RE_ID_EXAMPLES_ROOT}/{summary_type}/{id}-discharge-inputs.txt"
            # Set outs
            output_filename = f"{RESULTS_DIR}/{target_model}/{v_name}/{summary_type}/{id}-discharge-inputs.txt"
            baseline_output_file = f"{RESULTS_DIR}/{target_model}/{v_name}/{summary_type}_baseline/{id}-discharge-inputs.txt"
            icl_output_file = f"{RESULTS_DIR}/{target_model}/{v_name}/{summary_type}_in_context/{id}-discharge-inputs.txt"

            # Creating results dir
            if os.path.exists(output_filename) is False: # SNAG - always hits this because the output file will not exist
                print(f"File {input_filename} does not exist")
                print("Creating the results folder")
                os.makedirs(f"{RESULTS_DIR}/{target_model}/{v_name}/{summary_type}", exist_ok=True)
                os.makedirs(f"{RESULTS_DIR}/{target_model}/{v_name}/{summary_type}_baseline", exist_ok=True)
                os.makedirs(f"{RESULTS_DIR}/{target_model}/{v_name}/{summary_type}_in_context", exist_ok=True)

            print(f"Starting inference for {id}")
            file_input = read_file(input_filename)

            # if the output file already exists, skip it (inference already completed)
            if file_input is not None and os.path.exists(output_filename) is False:
                print('Privacy Baseline')
                # get a pseduo summary
                pseudo_trimmed_res = run_inference(instruction_main, file_input, model, tokenizer)
                # write pseudonimysed output
                write_to_file(output_filename, pseudo_trimmed_res)

            if file_input is not None and os.path.exists(baseline_output_file) is False:
                print('Instruction Baseline')
                # get a baseline (non-pseduo summary)
                baseline_trimmed_res = run_inference(instruction_baseline, file_input, model, tokenizer)
                # write baseline output
                write_to_file(baseline_output_file, baseline_trimmed_res)

            if file_input is not None and os.path.exists(icl_output_file) is False:
                print('ICL')
                # get a baseline (non-pseduo summary)
                # add an in-context example
                instruction_icl = instruction_icl.replace("[incontext_examples]", f"{fetch_example(summary_type)}")
                icl_trimmed_res = run_inference(instruction_icl, file_input, model, tokenizer)
                # write baseline output
                write_to_file(icl_output_file, icl_trimmed_res)
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
    run_all_inference(target_model=args.model, hadm_ids=hadm_ids, summary_type=args.task)
    print("Inference complete")


if __name__ == "__main__":
    main()
