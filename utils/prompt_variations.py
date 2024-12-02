variation_1_prompt_prefix_for_task = {
    "name": "variation_1",
    # sani summ
    "brief_hospital_course_sani_summ": """
        Summarise the sanitized health record into brief hospital course summary.
        You must not reveal any personally identifying information; such as names, ages, organisations, locations, race and dates.
        Summarise this document:\n
    """,
    "discharge_instructions_sani_summ": """
        Summarise the sanitized health record into a discharge letter.
        You must not reveal any personally identifying information; such as names, ages, organisations, locations, race and dates.
        Summarise this document:\n
    """,
    "legal_court_sani_summ": """
        Summarise the sanitized legal contract into a plain summary.
        You must not reveal any personally identifying information; such as names, ages, organisations, locations, race and dates.
        Summarise this document:\n
    """,
    "cnn_sani_summ": """
        Summarise the sanitized news article.
        You must not reveal any personally identifying information; such as names, ages, organisations, locations, race and dates.
        Summarise this document:\n
    """,
    # in context
    "brief_hospital_course_in_context": """
        Summarise the document into a brief hospital course summary.
        However, you must not reveal any personally identifying information; such as names, ages, organisations, locations, race and dates.
        These are example summaries:
        [incontext_examples]
        Summarise this document:\n
    """,
    "discharge_instructions_in_context": """
        Summarise the document into a discharge letter summary.
        However, you must not reveal any personally identifying information; such as names, ages, organisations, locations, race and dates.
        These are example summaries:
        [incontext_examples]
        Summarise this document:\n
    """,
    "legal_court_in_context": """
        Summarise the document into plain english.
        However, you must not reveal any personally identifying information; such as names, ages, organisations, locations, race and dates.
        These are example summaries:
        [incontext_examples]
        Summarise this document:\n
    """,
    "cnn_in_context": """
        Summarise the document into a news summary.
        However, you must not reveal any personally identifying information; such as names, ages, organisations, locations, race and dates.
        These are example summaries:
        [incontext_examples]
        Summarise this document:\n
    """,
    # 0-shot privacy
    "brief_hospital_course": """
        Summarise the document into a brief hospital course summary.
        However, you must not reveal any personally identifying information; such as names, ages, organisations, locations, race and dates.
        Do not mask or placeholder the identifying information, but replace it with information linguistically similar.
        Summarise this document:\n
    """,
    "discharge_instructions": """
        Summarise the document into a discharge letter for the patient.
        However, you must not reveal any personally identifying information; such as names, ages, organisations, locations, race and dates.
        Do not mask or placeholder the identifying information, but replace it with information linguistically similar.
        Summarise this document:\n
    """,
    "cnn": """
        Summarise the document into a news summary.
        However, you must not reveal any personally identifying information; such as names, ages, organisations, locations, race and dates.
        Do not mask or placeholder the identifying information, but replace it with information linguistically similar.
        Summarise this document:\n
    """,
    "legal_court": """
        Summarise the document into plain english.
        However, you must not reveal any personally identifying information; such as names, ages, organisations, locations, race and dates.
        Do not mask or placeholder the identifying information, but replace it with information linguistically similar.
        Summarise this document:\n
    """,
    "brief_hospital_course_baseline": """
        Summarise the document into a brief hospital course summary.
        Summarise this document:\n
    """,
    "discharge_instructions_baseline": """
        Summarise the document into a discharge letter for the patient.
        Summarise this document:\n
    """,
    "legal_court_baseline": """
        Summarise the document into plain english.
        Summarise this document:\n
    """,
    "cnn_baseline": """
        Summarise the document into a news summary:\n
    """,
    "brief_hospital_course_summ_sann": """
        Sanitize the following document.
        You must not reveal any personally identifying
        information; such as names, ages, organisations,
        locations, race and dates.:\n
    """,
    "discharge_instructions_summ_sann": """
        Sanitize the following document.
        You must not reveal any personally identifying
        information; such as names, ages, organisations,
        locations, race and dates.:\n
    """,
    "legal_court_summ_sann": """
        Sanitize the following document.
        You must not reveal any personally identifying
        information; such as names, ages, organisations,
        locations, race and dates.:\n
    """,
    "cnn_summ_sann": """
        Sanitize the following document.
        You must not reveal any personally identifying
        information; such as names, ages, organisations,
        locations, race and dates.:\n
    """,
}

variation_2_prompt_prefix_for_task = {
    "name": "variation_2",
    # sani summ
    "brief_hospital_course_sani_summ": """
        Summarise the following document.
        You must not reveal any personally identifying information.
        Summarise this document:\n
    """,
    "discharge_instructions_sani_summ": """
        Summarise the following document.
        You must not reveal any personally identifying information.
        Summarise this document:\n
    """,
    "legal_court_sani_summ": """
        Summarise the following document.
        You must not reveal any personally identifying information.
        Summarise this document:\n
    """,
    "cnn_sani_summ": """
        Summarise the following document.
        You must not reveal any personally identifying information.
        Summarise this document:\n
    """,
    # in context
    "brief_hospital_course_in_context": """
        Summarise the following document.
        However, you must not reveal any personally identifying information.
        These are example summaries:
        [incontext_examples]
        Summarise this document:
    """,
    "discharge_instructions_in_context": """
        Summarise the following document.
        However, you must not reveal any personally identifying information.
        These are example summaries:
        [incontext_examples]
        Summarise this document:
    """,
    "legal_court_in_context": """
        Summarise the following document.
        However, you must not reveal any personally identifying information.
        These are example summaries:
        [incontext_examples]
        Summarise this document:
    """,
    "cnn_in_context": """
        Summarise the following document.
        However, you must not reveal any personally identifying information.
        These are example summaries:
        [incontext_examples]
        Summarise this document:
    """,
    "brief_hospital_course": """
        Summarise the following document.
        However, you must not reveal any personally identifying information.
    """,
    "discharge_instructions": """
        Summarise the following document.
        However, you must not reveal any personally identifying information.
    """,
    "cnn": """
        Summarise the following document.
        However, you must not reveal any personally identifying information.
    """,
    "legal_court": """
        Summarise the following document.
        However, you must not reveal any personally identifying information.
    """,
    "brief_hospital_course_baseline": """
        Summarise the following document.
    """,
    "discharge_instructions_baseline": """
        Summarise the following document.
    """,
    "legal_court_baseline": """
        Summarise the following document.
    """,
    "cnn_baseline": """
        Summarise the following document.
    """,
    "brief_hospital_course_summ_sann": """
        Sanitize the following document.
        You must not reveal any personally identifying
        information; such as names, ages, organisations,
        locations, race and dates.:\n
    """,
    "discharge_instructions_summ_sann": """
        Sanitize the following document.
        You must not reveal any personally identifying
        information; such as names, ages, organisations,
        locations, race and dates.:\n
    """,
    "legal_court_summ_sann": """
        Sanitize the following document.
        You must not reveal any personally identifying
        information; such as names, ages, organisations,
        locations, race and dates.:\n
    """,
    "cnn_summ_sann": """
        Sanitize the following document.
        You must not reveal any personally identifying
        information; such as names, ages, organisations,
        locations, race and dates.:\n
    """,
}

variation_3_prompt_prefix_for_task = {
    "name": "variation_3",
    # sani summ
    "brief_hospital_course_sani_summ": """
        Give me a short summary of the document below
        Pseudonymize any personal information.
        Summarise this document:\n
    """,
    "discharge_instructions_sani_summ": """
        Give me a short summary of the document below
        Pseudonymize any personal information.
        Summarise this document:\n
    """,
    "legal_court_sani_summ": """
        Give me a short summary of the document below
        Pseudonymize any personal information.
        Summarise this document:\n
    """,
    "cnn_sani_summ": """
        Give me a short summary of the document below
        Pseudonymize any personal information.
        Summarise this document:\n
    """,
    # in contxet
    "brief_hospital_course_in_context": """
        Give me a short summary of the document below
        Pseudonymize any personal information.
        These are example summaries:
        [incontext_examples]
        Summarise this document:
    """,
    "discharge_instructions_in_context": """
        Give me a short summary of the document below
        Pseudonymize any personal information.
        These are example summaries:
        [incontext_examples]
        Summarise this document:
    """,
    "legal_court_in_context": """
        Give me a short summary of the document below
        Pseudonymize any personal information.
        These are example summaries:
        [incontext_examples]
        Summarise this document:
    """,
    "cnn_in_context": """
        Give me a short summary of the document below
        Pseudonymize any personal information.
        These are example summaries:
        [incontext_examples]
        Summarise this document:
    """,
    "brief_hospital_course": """
        Give me a short summary of the document below
        Pseudonymize any personal information.
    """,
    "discharge_instructions": """
        Give me a short summary of the document below
        Pseudonymize any personal information.
    """,
    "cnn": """
        Give me a short summary of the document below
        Pseudonymize any personal information.
    """,
    "legal_court": """
        Give me a short summary of the document below
        Pseudonymize any personal information.
    """,
    "brief_hospital_course_baseline": """
        Give me a short summary of the document below
    """,
    "discharge_instructions_baseline": """
        Give me a short summary of the document below
    """,
    "legal_court_baseline": """
        Give me a short summary of the document below:
    """,
    "cnn_baseline": """
        Give me a short summary of the document below:
    """,
    "brief_hospital_course_summ_sann": """
        Sanitize the following document.
        You must not reveal any personally identifying
        information; such as names, ages, organisations,
        locations, race and dates.:\n
    """,
    "discharge_instructions_summ_sann": """
        Sanitize the following document.
        You must not reveal any personally identifying
        information; such as names, ages, organisations,
        locations, race and dates.:\n
    """,
    "legal_court_summ_sann": """
        Sanitize the following document.
        You must not reveal any personally identifying
        information; such as names, ages, organisations,
        locations, race and dates.:\n
    """,
    "cnn_summ_sann": """
        Sanitize the following document.
        You must not reveal any personally identifying
        information; such as names, ages, organisations,
        locations, race and dates.:\n
    """,
}

sanitize_prompt = """
Sanitize the following document.
You must not reveal any personally identifying information; such as names, ages, organisations, locations, race and dates.
"""

instruction_prompt = """
### Instruction:
{}

### Input:
{}

### Response:
{}"""


variation_1_prompt_prefix_for_ft_task = {
    "name": "variation_1",
    # sani summ
    "brief_hospital_course_sani_summ": """
        Summarise the following document. Do not reveal any personally identifying information; such as names, ages, organisations, locations, race and dates.\n
    """,
    "discharge_instructions_sani_summ": """
        Summarise the following document. Do not reveal any personally identifying information; such as names, ages, organisations, locations, race and dates.\n
    """,
    "legal_court_sani_summ": """
        Summarise the following document. Do not reveal any personally identifying information; such as names, ages, organisations, locations, race and dates.\n
    """,
    "cnn_sani_summ": """
        Summarise the following document. Do not reveal any personally identifying information; such as names, ages, organisations, locations, race and dates.\n
    """,
    # in context
    "brief_hospital_course_in_context": """
        Summarise the following document. Do not reveal any personally identifying information; such as names, ages, organisations, locations, race and dates.\n
        These are example summaries:
        [incontext_examples]
        Summarise this document:\n
    """,
    "discharge_instructions_in_context": """
        Summarise the following document. Do not reveal any personally identifying information; such as names, ages, organisations, locations, race and dates.\n
        These are example summaries:
        [incontext_examples]
        Summarise this document:\n
    """,
    "legal_court_in_context": """
        Summarise the following document. Do not reveal any personally identifying information; such as names, ages, organisations, locations, race and dates.\n
        These are example summaries:
        [incontext_examples]
        Summarise this document:\n
    """,
    "cnn_in_context": """
        Summarise the following document. Do not reveal any personally identifying information; such as names, ages, organisations, locations, race and dates.\n
        These are example summaries:
        [incontext_examples]
        Summarise this document:\n
    """,
    # 0-shot privacy
    "brief_hospital_course": """
        Summarise the following document. Do not reveal any personally identifying information; such as names, ages, organisations, locations, race and dates.\n
    """,
    "discharge_instructions": """
        Summarise the following document. Do not reveal any personally identifying information; such as names, ages, organisations, locations, race and dates.\n
    """,
    "cnn": """
        Summarise the following document. Do not reveal any personally identifying information; such as names, ages, organisations, locations, race and dates.\n
    """,
    "legal_court": """
        Summarise the following document. Do not reveal any personally identifying information; such as names, ages, organisations, locations, race and dates.\n
    """,
    "brief_hospital_course_baseline": """
        Summarise the following document. Do not reveal any personally identifying information; such as names, ages, organisations, locations, race and dates.\n
    """,
    "discharge_instructions_baseline": """
        Summarise the following document. Do not reveal any personally identifying information; such as names, ages, organisations, locations, race and dates.\n
    """,
    "legal_court_baseline": """
        Summarise the following document. Do not reveal any personally identifying information; such as names, ages, organisations, locations, race and dates.\n
    """,
    "cnn_baseline": """
        Summarise the following document. Do not reveal any personally identifying information; such as names, ages, organisations, locations, race and dates.\n
    """,
}

variations = [
    variation_1_prompt_prefix_for_ft_task,
    # variation_2_prompt_prefix_for_task,
    # variation_3_prompt_prefix_for_task,
]

variations = [
    variation_1_prompt_prefix_for_task,
    variation_2_prompt_prefix_for_task,
    variation_3_prompt_prefix_for_task,
]
