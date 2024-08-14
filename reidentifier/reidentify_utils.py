from glob import glob
import os
import random
import names
from dateutil.relativedelta import relativedelta
from dateutil import parser
import datetime

from utils.dataset_utils import fetch_admission_info


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
    return f"{names.get_full_name()}"


def generate_random_clinician_name():
    return f"Dr. {names.get_full_name()}"


def generate_random_date(discharge_date):
    return "2140-06-06"


def generate_intervention_date(discharge_date):
    """
    Generate a date between 1 and 3 days before the discharge date
    """
    day = random.randint(1, 7)
    parsed_dt = parser.parse(discharge_date)
    new_date = parsed_dt - relativedelta(days=day)
    return new_date.strftime("%Y-%m-%d")


def generate_birth_date(age, discharge_date):
    """
    Generate birth date based on age and discharge date
    """
    parsed_dt = parser.parse(discharge_date)
    return parsed_dt - relativedelta(years=age)


def generate_random_location():
    return "New York Hospital"


def remove_extra_redactions(discharge_data):
    """Remove extra redactions"""
    discharge_data = discharge_data.replace("(___)", "")
    discharge_data = discharge_data.replace("___", " ")
    return discharge_data


def remove_extra_piis(discharge_data):
    """Remove extra redactions"""
    for pii in [
        "year-old",
        "y/o",
    ]:
        discharge_data = discharge_data.replace(pii, " ")
    return discharge_data


def generate_random_profile(hadm_id):
    admission_info = fetch_admission_info(hadm_id)
    age = generate_random_adult_age()
    out_date = (
        list(admission_info["outtime"])[0]
        if len(admission_info["outtime"].values) > 0
        else generate_random_date()
    )
    return {
        "name": generate_random_name(),
        "clinician_name": generate_random_clinician_name(),
        "age": age,
        "in_date": (
            list(admission_info["intime"])[0]
            if len(admission_info["intime"].values) > 0
            else generate_random_date()
        ),
        "out_date": out_date,
        "birth_date": generate_birth_date(age, out_date),
        "location": generate_random_location(),
        "intervention_date": generate_intervention_date(out_date),
    }
