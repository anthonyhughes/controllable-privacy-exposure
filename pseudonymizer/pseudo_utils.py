import re


def deidentify_text(text, deidentification_dict):
    """
    Replace deidentified placeholders in the text with corresponding terms.

    Parameters:
    - text: The input text containing placeholders (e.g., 'GPE0', 'DATE0').
    - deidentification_dict: A dictionary where keys are placeholders (e.g., 'GPE0', 'DATE0')
                             and values are the specific terms (e.g., 'organization', 'date')
                             to replace them with.

    Returns:
    - The deidentified text with placeholders replaced by specific terms.
    """
    # Iterate through the dictionary and replace placeholders in the text
    for placeholder, replacement in deidentification_dict.items():
        # Use regex to replace whole word occurrences of the placeholder
        text = re.sub(rf"\b{placeholder}\d*\w*\b", replacement, text)

    return text


def sanitize_text(text, deidentification_dict):
    """
    Replace deidentified placeholders in the text with corresponding terms.

    Parameters:
    - text: The input text containing placeholders (e.g., 'GPE0', 'DATE0').
    - deidentification_dict: A dictionary where keys are placeholders (e.g., 'GPE0', 'DATE0')
                             and values are the specific terms (e.g., 'organization', 'date')
                             to replace them with.

    Returns:
    - The deidentified text with placeholders replaced by specific terms.
    """
    # Iterate through the dictionary and replace placeholders in the text
    for placeholder, replacement in deidentification_dict.items():
        # Use regex to replace whole word occurrences of the placeholder
        text = re.sub(rf"\b{placeholder}\d*\w*\b", "___", text)

    return text