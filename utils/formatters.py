def convert_to_llama_chat_template(message):
    return f"""<s>[INST] <<SYS>> You are a helpful assistant that summarizes truthfully any document given. <</SYS>> {message} [/INST]
"""


llama31_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{}<|eot_id|><|start_header_id|>user<|end_header_id|>

{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{}<|eot_id|>
"""
