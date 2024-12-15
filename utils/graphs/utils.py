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
        "discharge_instructions",
        "legal_court",
        "cnn",
    ]
    return clean_label(target_tasks[target_task_idx])


def clean_metric(metric):
    if metric == "Bertscore" or metric == "bertscore":
        return "BERTScore"
    elif (
        metric == "RougeL"
        or metric == "rougeL"
        or metric == "rougeLsum"
        or metric == "rougel"
        or metric == "roguel"
    ):
        return "Rouge-L"
    elif metric == "Rouge1" or metric == "rouge1":
        return "Rouge-1"
    elif metric == "tpr" or metric == "TPR":
        return "True Positive Rate"
    elif metric == "fpr":
        return "False Positive Rate"
    elif metric == "ptr" or metric == "PTR":
        return "Privacy Token Ratio"
    elif metric == "ldr" or metric == "LDR":
        return "Leaked Document Ratio"
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
    return [f"Variant {i+1}" for i in range(len(variations))]


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
        return "Anonymize & Summarize"
    else:
        return "0-Shot"


def micro_averaged_fpr(false_positives, true_negatives):
    """
    Calculate Micro-Averaged False Positive Rate (FPR).

    Parameters:
    - false_positives (list): A list of false positives for each class.
    - true_negatives (list): A list of true negatives for each class.

    Returns:
    - float: Micro-Averaged False Positive Rate (FPR).
    """
    if not false_positives or not true_negatives:
        raise ValueError(
            "Both false_positives and true_negatives lists must be provided."
        )
    if len(false_positives) != len(true_negatives):
        raise ValueError(
            "Lists false_positives and true_negatives must be of the same length."
        )

    # Sum across all classes
    total_fp = sum(false_positives)
    total_tn = sum(true_negatives)

    # Calculate Micro-Averaged FPR
    micro_fpr = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0.0

    return micro_fpr
