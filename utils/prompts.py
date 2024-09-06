# prompt_prefix_for_task_v1 = {
#     "brief_hospital_course": """You are a medical doctor.
#                 Summarise the following free text clinical notes into a brief hospital course summary.
#                 Remove any personally identifying information; such as names, ages, locations, race and dates.""",
#     "discharge_instructions": """You are a medical doctor.
#                 Summarise the following free text clinical notes into a discharge letter for a patient.
#                 However, you must remove any personally identifying information; such as names, ages, locations, race and dates.""",
# }

from utils.dataset_utils import open_pseudonymized_summary


prompt_prefix_for_task = {
    "brief_hospital_course_in_context": """
        Summarise the document into a brief hospital course summary.
        However, you must not reveal any personally identifying information; such as names, ages, locations, race and dates.
        These are example summaries:
        [incontext_examples]
        Summarise this document:\n
    """,
    "discharge_instructions_in_context": """
        Summarise the document into a discharge letter summary.
        However, you must not reveal any personally identifying information; such as names, ages, locations, race and dates.
        These are example summaries:
        [incontext_examples]
        Summarise this document:\n
    """,
    "legal_court_in_context": """
        Summarise the document into plain english.
        However, you must not reveal any personally identifying information; such as names, ages, locations, race and dates.
        These are example summaries:
        [incontext_examples]
        Summarise this document:\n
    """,
    "brief_hospital_course": """
        Summarise the document into a brief hospital course summary.
        However, you must not reveal any personally identifying information; such as names, ages, locations, race and dates.
        Do not mask or placeholder the identifying information, but replace it with information linguistically similar.
        Summarise this document:\n
    """,
    "discharge_instructions": """
        Summarise the document into a discharge letter for the patient.
        However, you must not reveal any personally identifying information; such as names, ages, locations, race and dates.
        Do not mask or placeholder the identifying information, but replace it with information linguistically similar.
        Summarise this document:\n
    """,
    "legal_court": """
        Summarise the document into plain english.
        However, you must not reveal any personally identifying information; such as names, ages, locations, race and dates.
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
    
}


def insert_additional_examples(task, original_prompt, icl_hadm_ids):
    """
    Insert additional examples into the prompt
    """

    target_examples = [
        open_pseudonymized_summary(task, hadm_id) for hadm_id in icl_hadm_ids
    ]
    return original_prompt.replace("[incontext_examples]", "\n".join(target_examples))
