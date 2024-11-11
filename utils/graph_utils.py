import json
import matplotlib.pyplot as plt
import numpy as np
from constants import (
    EVAL_MODELS_REAL,
    FINAL_REID_RESULTS_DIR,
    PRIVACY_RESULTS_DIR,
)
import seaborn as sns
import pandas as pd

from utils.graphs.heatmaps import plot_heat_map


def gen_graphs():
    """
    Generate graphs for the re-identification results
    """
    files = [
        "claude-3-5-sonnet-20240620-reidentification_results-20241007-093020.json",
        "gpt-4o-mini-reidentification_results-20241008-093250.json",
        "llama-3-8b-Instruct-bnb-4bit-reidentification_results-20241008-095559.json",
        "Meta-Llama-3.1-70B-Instruct-bnb-4bit-reidentification_results-20241008-092100.json",
        "mistral-7b-instruct-v0.3-bnb-4bit-reidentification_results-20241008-091200.json",
    ]

    heat_data = np.zeros((len(files), 6))
    tasks = []
    for i, file in enumerate(files):
        with open(f"{FINAL_REID_RESULTS_DIR}/{file}", "r") as f:
            data = json.load(f)
            for j, task in enumerate(data.keys()):
                di_icl = data[task]
                for clazz in di_icl.keys():
                    heat_data[i][j] += di_icl[clazz]["tp"]

    tasks = [
        "DI - 0-Shot",
        "DI - 1-Shot",
        "DI - S&S",
        "BHC - 0-Shot",
        "BHC - 1-Shot",
        "BHC - S&S",
    ]
    plot_heat_map(EVAL_MODELS_REAL, tasks, heat_data)
    print("Graphs generated successfully!")
