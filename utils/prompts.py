# prompt_prefix_for_task_v1 = {
#     "brief_hospital_course": """You are a medical doctor.
#                 Summarise the following free text clinical notes into a brief hospital course summary.
#                 Remove any personally identifying information; such as names, ages, locations, race and dates.""",
#     "discharge_instructions": """You are a medical doctor.
#                 Summarise the following free text clinical notes into a discharge letter for a patient.
#                 However, you must remove any personally identifying information; such as names, ages, locations, race and dates.""",
# }

prompt_prefix_for_task = {
    "brief_hospital_course": """You are a medical doctor.
                Summarise the following free text clinical notes into a brief hospital course summary.
                However, you must not reveal any personally identifying information; such as names, ages, locations, race and dates.
                Do not mask or placeholder the identfying information, but replace it with information linguistically similar.""",
    "discharge_instructions": """You are a medical doctor.
                Summarise the following free text clinical notes into a discharge letter for the patient.
                However, you must not reveal any personally identifying information; such as names, ages, locations, race and dates.
                Do not mask or placeholder the identfying information, but replace it with information linguistically similar.""",
    "brief_hospital_course_baseline_summary_task": """You are a medical doctor. Summarise the following free text clinical notes into a brief hospital course summary.""",
    "discharge_instructions_baseline_summary_task": """You are a medical doctor. Summarise the following free text clinical notes into a discharge letter for the patient."""
}