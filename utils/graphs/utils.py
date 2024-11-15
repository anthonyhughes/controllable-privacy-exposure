def clean_label(label) -> str:
    """
    Clean the label for the task
    """
    if label == "cnn":
        return "CNN/DailyMail"
    return label.replace("_", " ").title()