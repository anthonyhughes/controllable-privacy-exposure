# This pipeline requires manually install of unsloth
# pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"
import argparse
from unsloth import FastLanguageModel
from constants import (
    EVAL_MODELS,
    MAX_TOKENS,
)
from datasets import load_dataset
from utils.formatters import llama31_prompt
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

_EOS_TOKEN = None


def get_model(target_model):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/" + target_model,
        max_seq_length=MAX_TOKENS,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = llama31_prompt.format(instruction, input, output)
        texts.append(text)
    return {
        "text": texts,
    }


def load_ft_data(training_data_file):
    training_data_loc = f"./data/fine-tuning/{training_data_file}"
    dataset = load_dataset("csv", data_files={"train": [training_data_loc]})
    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
    )
    return dataset


def run_fine_tuning(target_model, training_data_file):
    # for each summary type
    model, tokenizer = get_model(target_model=target_model)
    global _EOS_TOKEN
    _EOS_TOKEN = tokenizer.eos_token
    dataset = load_ft_data(training_data_file)
    # print(dataset["train"][10]["text"])
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        dataset_text_field="text",
        max_seq_length=MAX_TOKENS,
        dataset_num_proc=2,
        packing=False,  # Can make training 5x faster for short sequences.
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=1,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",
        ),
    )
    trainer.train()
    version = "v1"
    model.save_pretrained(
        f"./data/ft_models/llamonymous-3-70b-bnb-4bit/{version}/lora_model"
    )
    tokenizer.save_pretrained(
        f"./data/ft_models/llamonymous-3-70b-bnb-4bit/{version}/lora_model"
    )


def main():
    print("Starting inference")
    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        help="Choose a target model for fine-tuning",
        default=EVAL_MODELS[4],
        choices=EVAL_MODELS,
    )
    parser.add_argument(
        "-t",
        "--training_data_file",
        help="Choose a target training set",
    )
    args = parser.parse_args()

    run_fine_tuning(target_model=args.model, training_data_file=args.training_data_file)
    print("Inference complete")


if __name__ == "__main__":
    main()
