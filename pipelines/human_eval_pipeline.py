import argparse
import csv
import os
from anthropic import Anthropic
from openai import OpenAI
from pydeidentify import Deidentifier

from utils.pii_eval import fetch_total_pii_count


def openai_gold_inference(prompt, ehr, model):
    print(f'Running gold for {model}')
    open_ai_client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    """Run the openai query"""
    prompt = {
        "role": "user",
        "content": f"""
                ### Instruction:
                {prompt}

                ### Input:
                {ehr}

                ### Response:
            """,
    }
    chat_completion = open_ai_client.chat.completions.create(
        messages=[prompt],
        model=model,
    )
    summary = chat_completion.choices[0].message.content
    return summary


def anthropic_gold_inference(prompt, ehr, model):
    client = Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )
    prompt = {
        "role": "user",
        "content": f"""
                ### Instruction:
                {prompt}

                ### Input:
                {ehr}

                ### Response:
            """,
    }
    chat_completion = client.messages.create(
        max_tokens=2048,
        messages=[prompt],
        model=model,
    )
    summary = chat_completion.content
    return summary[0].text


def save_result(model, result, type_res, id):
    with open(f"data/gold/{model}/{type_res}/{id}.txt", "r") as f:
        f.write(f"{result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    d = Deidentifier()
    models = ["gpt-4o-mini", "claude-3-5-sonnet-20240620"]
    for model in models:
        # all_counts = 0    
        with open("data/human_eval/human_priv_sums_v2.csv") as f:
            csv_reader = csv.reader(f, quotechar='"')
            # print(csv_reader.__sizeof__())
            print(f"Reading file: {f.name}")
            for cols in csv_reader:
                input_doc = cols[0]
                name = input_doc[6:25]
                if os.path.exists(f"data/gold/{model}/reid/{name}.txt"):
                    print(f"Skipping previous inference {model} {name}")
                    continue
                if input_doc.startswith("Name:") == False:
                    print(f"Skipping file {name}")
                    continue
                print('Name:', name)
                scrubbed_info = d.deidentify(input_doc)
                scrubbed_text = scrubbed_info.text
                # original_private_token_counts = fetch_total_pii_count(scrubbed_text)
                # all_counts = all_counts + original_private_token_counts
                # print(f"Original private token counts: {original_private_token_counts}")
                # if original_private_token_counts > 0:
                #     print(f"Scrubbed text: {scrubbed_text.counts}")
                #     print(f"Scrubbed text: {scrubbed_text.text}")
                #     print(f"Scrubbed text: {scrubbed_text.encode_mapping}")
                prompt = """Summarise the document into a brief hospital course summary.
            However, you must not reveal any personally identifying information; such as names, ages, organisations, locations, race and dates.
            These are example summaries:
            [incontext_examples]
            Summarise this document:\n"""
                if model == "gpt-4o-mini":
                    reid_result = openai_gold_inference(prompt, input_doc, model)
                    scrubbed_result = openai_gold_inference(prompt, scrubbed_text, model)
                elif model == "claude-3-5-sonnet-20240620":
                    reid_result = anthropic_gold_inference(prompt, input_doc, model)
                    scrubbed_result = anthropic_gold_inference(prompt, scrubbed_text, model)

                save_result(model, reid_result, "reid", name)
                save_result(model, scrubbed_result, "scrubbed", name)
                print(f"Done for {name}")

    # print(f"Total private token counts: {all_counts}")
