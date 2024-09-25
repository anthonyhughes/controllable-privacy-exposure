DATA_ROOT = "data/"
EXAMPLES_ROOT = DATA_ROOT + "examples/"
ICL_EXAMPLES_ROOT = DATA_ROOT + "icl_examples/"
PSEUDO_TARGETS_ROOT = DATA_ROOT + "pseudonymized_targets/"
RE_ID_TARGETS_ROOT = DATA_ROOT + "re_identified_targets"
RE_ID_EXAMPLES_ROOT = DATA_ROOT + "re_identified_examples/"
LEGAL_EXAMPLES_ROOT = EXAMPLES_ROOT + "legal_court/"
PSEUDO_PROFILES_LOCATION = f"{DATA_ROOT}/pseudo-profiles.json"
DISCHARGE_ME_ROOT = (
    DATA_ROOT
    + "discharge-me-bionlp-acl24-shared-task-on-streamlining-discharge-documentation-1.3/"
)
TRAIN_DISCHARGE_ME = DISCHARGE_ME_ROOT + "train/"
SUMMARY_TYPES = ["brief_hospital_course", "discharge_instructions", "legal_court", "cnn"]
ALT_SUMMARY_TYPES = ["legal_court"]
EXAMPLE_ADMISSION_IDS = [
    22343752,
    25404430,
    20522931,
    22907047,
    26326405,
    26818922,
    23364124,
    23936893,
    27771974,
    20910785,
]
RESULTS_DIR = DATA_ROOT + "results"
UTILITY_RESULTS_DIR = DATA_ROOT + "utility_results"
PRIVACY_RESULTS_DIR = DATA_ROOT + "privacy_results"
BATCH_RESULTS_DIR = DATA_ROOT + "batch_results"
BATCH_JOBS_DIR = DATA_ROOT + "batch_jobs"
MODELS = [
    "gpt-4o-mini",
    "mistral-instruct-7b",
]
EVAL_MODELS = [
    "gpt-4o-mini",
    "mistral-7b-instruct-v0.3-bnb-4bit",
    "llama-3-8b-Instruct-bnb-4bit",
    "claude-3-5-sonnet-20240620",
    "Meta-Llama-3.1-70B-Instruct-bnb-4bit"
    # "llama3.1:70b"
]
EVAL_TYPES = ["utility", "privacy", "reidentification", "all"]
BASELINE_SUMMARY_TASK = "_baseline"
BATCH_FLAGS = ["batch", "check", "retrieve"]
PRIV_SUMMARY_TASK = ""
IN_CONTEXT_SUMMARY_TASK = "_in_context"
TASK_SUFFIXES = [BASELINE_SUMMARY_TASK, PRIV_SUMMARY_TASK, IN_CONTEXT_SUMMARY_TASK]
METRICS = ["rouge1", "rouge2", "rougeL", "bertscore"]
PSUEDO_LIBS = ["spacy", "flair", "bert-deid"]
DEIDENTIFICATION_DICT = {
    "GPE": "the location",
    "DATE": "the date",
    "LOC": "the location",
    "NAME": "the person",
    "PERSON": "A person",
    "ORG": "the organisation",
    "FAC": "the location"
}
