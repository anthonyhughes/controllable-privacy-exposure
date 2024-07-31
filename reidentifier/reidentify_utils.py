from glob import glob
import os
import random


def fetch_file_names(path, type, hadm_id="*"):
    files = list(glob(os.path.join(f"{path}", f"{hadm_id}-{type}.txt")))
    return files


def load_file(file):
    with open(file, "r") as f:
        return f.read()


def generate_random_unit_number(start=1, end=100):
    """Generate random unit number between start and end"""
    return random.randint(start, end)


def generate_random_week_number(start=1, end=52):
    """Generate random week number between start and end"""
    return random.randint(start, end)


def generate_random_adult_age(start=18, end=100):
    """Generate random adult age between start and end"""
    return random.randint(start, end)


def generate_random_name():
    return "John Doe"


def generate_random_date():
    return "01/01/2021"


def generate_random_location():
    return "New York Hospital"


def remove_extra_redactions(discharge_data):
    """Remove extra redactions"""
    discharge_data = discharge_data.replace("(___)", "")
    discharge_data = discharge_data.replace("___", " ")    
    return discharge_data

def remove_extra_piis(discharge_data):
    """Remove extra redactions"""
    for pii in ["year-old", "y/o",]:
        discharge_data = discharge_data.replace(pii, " ")
    return discharge_data
