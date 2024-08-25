# prompt_prefix_for_task_v1 = {
#     "brief_hospital_course": """You are a medical doctor.
#                 Summarise the following free text clinical notes into a brief hospital course summary.
#                 Remove any personally identifying information; such as names, ages, locations, race and dates.""",
#     "discharge_instructions": """You are a medical doctor.
#                 Summarise the following free text clinical notes into a discharge letter for a patient.
#                 However, you must remove any personally identifying information; such as names, ages, locations, race and dates.""",
# }

prompt_prefix_for_task = {
    "brief_hospital_course_in_context_summary_task": """
        Summarise the following document into a brief hospital course summary.
        However, you must not reveal any personally identifying information; such as names, ages, locations, race and dates.
        These are example summaries:
        [incontext_examples]
        Summarise this document:\n
    """,
    "discharge_instructions_in_context_summary_task": """
        Summarise the following document into a discharge letter summary.
        However, you must not reveal any personally identifying information; such as names, ages, locations, race and dates.
        These are example summaries:
        [incontext_examples]
        Summarise this document:\n
    """,
    "brief_hospital_course": """
        Summarise the following free text clinical notes into a brief hospital course summary.
        However, you must not reveal any personally identifying information; such as names, ages, locations, race and dates.
        Do not mask or placeholder the identifying information, but replace it with information linguistically similar.
    """,
    "discharge_instructions": """
        Summarise the following free text clinical notes into a discharge letter for the patient.
        However, you must not reveal any personally identifying information; such as names, ages, locations, race and dates.
        Do not mask or placeholder the identifying information, but replace it with information linguistically similar.
    """,
    "brief_hospital_course_baseline_summary_task": """
        Summarise the following free text clinical notes into a brief hospital course summary.
    """,
    "discharge_instructions_baseline_summary_task": """
        Summarise the following free text clinical notes into a discharge letter for the patient.
    """,
}


def insert_additional_examples(original_prompt, examples, k):
    """
    Insert additional examples into the prompt
    """    
    return original_prompt.replace("[incontext_examples]", "\n".join(examples))