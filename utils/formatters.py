def convert_to_llama_chat_template(message):
    return f"""<s>[INST] <<SYS>> You are a helpful assistant that summarizes truthfully any document given. <</SYS>> {message} [/INST]
"""
