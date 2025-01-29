import os
from transformers import AutoTokenizer

def get_tokenizer_and_context(model_name):
    """Returns the tokenizer and context length for a given model name."""
    model_to_context = {
        "llama-8b": ("meta-llama/Llama-3.2-1B", 8192),
        "mistral-7b": ("mistralai/Mistral-7B-Instruct-v0.3", 8192),
    }
    if model_name not in model_to_context:
        raise ValueError(f"Unknown model name: {model_name}")

    return model_to_context[model_name]

def count_tokens(file_path, tokenizer):
    """Counts the number of tokens in a file using the provided tokenizer."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    tokens = tokenizer(text).input_ids
    return len(tokens)

def identify_exceeding_files(directory, model_name):
    """Identifies files in a directory that exceed the context length window for the given model."""
    tokenizer_name, context_length = get_tokenizer_and_context(model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_auth_token=True)

    exceeding_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                token_count = count_tokens(file_path, tokenizer)
                if token_count > context_length:
                    exceeding_files.append((file_path, token_count))
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    return exceeding_files

def main():
    model_name = input("Enter the model name (e.g., llama-8b, mistral-7b): ")

    try:
        exceeding_files = identify_exceeding_files("data/re_identified_examples/brief_hospital_course", model_name)
        if exceeding_files:
            print("The following files exceed the context length window:")
            print(f"File count = {len(exceeding_files)}")
            for file_path, token_count in exceeding_files:
                print(f"{file_path} - {token_count} tokens")
        else:
            print("No files exceed the context length window.")
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()
