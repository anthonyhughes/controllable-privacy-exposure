from constants import EVAL_MODELS_REAL_MAPPING


def clean_label(label) -> str:
    """
    Clean the label for the task
    """
    if label == "cnn":
        return "CNN/DailyMail"
    return label.replace("_", " ").title()


def clean_model_name(model_name):
    return EVAL_MODELS_REAL_MAPPING[model_name]


def clean_metric(metric):
    if metric == "Bertscore":
        return "BERTScore"
    elif metric == "RougeL":
        return "Rouge-L"
    else:
        return metric


def clean_privacy_metric(priv_metric):
    if priv_metric == "pii_document_percentage":
        return "Leaked Document Ratio"
    elif priv_metric == "private_token_ratio":
        return "Private Token Ratio"
    else:
        return priv_metric
