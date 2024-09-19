variation_1_prompt_prefix_for_task = {
    "name": "variation_1",
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
}

variation_2_prompt_prefix_for_task = {
    "name": "variation_2",
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
}

variation_3_prompt_prefix_for_task = {
    "name": "variation_3",
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
}

instruction_prompt = """
### Instruction:
{}

### Input:
{}

### Response:
{}"""


variations = [
    variation_1_prompt_prefix_for_task,
    variation_2_prompt_prefix_for_task,
    variation_3_prompt_prefix_for_task,
]
