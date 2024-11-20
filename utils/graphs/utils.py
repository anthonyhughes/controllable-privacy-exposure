from constants import EVAL_MODELS_REAL_MAPPING


def clean_label(label) -> str:
    """
    Clean the label for the task
    """
    if label == "cnn":
        return "CNN/DailyMail"
    if label == "legal_court":
        return "Legal Contracts"
    return label.replace("_", " ").title()


def clean_model_name(model_name):
    return EVAL_MODELS_REAL_MAPPING[model_name]


def fetch_clean_dataset_name(target_task_idx):
    target_tasks = [
        "brief_hospital_course",
        "cnn",
        "discharge_instructions",
        "legal_court",
    ]
    return clean_label(target_tasks[target_task_idx])


def clean_metric(metric):
    if metric == "Bertscore" or metric == "bertscore":
        return "BERTScore"
    elif metric == "RougeL" or metric == "rougeL":
        return "Rouge-L"
    elif metric == "Rouge1" or metric == "rouge1":
        return "Rouge-1"
    elif metric == "tpr":
        return "True Positive Rate"
    elif metric == "fpr":
        return "False Positive Rate"
    elif metric == "ptr":
        return "Privacy Token Ratio"
    else:
        return metric


def clean_privacy_metric(priv_metric):
    if priv_metric == "pii_document_percentage":
        return "Leaked Document Ratio"
    elif priv_metric == "private_token_ratio":
        return "Private Token Ratio"
    else:
        return priv_metric


def clean_variations(variations):
    return [f"Variation {i+1}" for i in range(len(variations))]


def clean_property(pii_property):
    if pii_property == "PERSON" or pii_property == "names":
        return "Names"
    elif pii_property == "DATE" or pii_property == "dates":
        return "Dates"
    elif pii_property == "ORG" or pii_property == "org":
        return "Organizations"
    else:
        return "All PII"

def clean_task_suffix(task_suffix):
    if task_suffix == "_in_context":
        return "1-Shot"
    elif task_suffix == "_sani_summ":
        return "Sanitized & Summarize"
    else:
        return "0-Shot"