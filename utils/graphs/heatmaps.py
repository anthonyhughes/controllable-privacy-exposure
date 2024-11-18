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

from utils.graphs.utils import clean_label, clean_model_name, clean_task_suffix


def plot_heat_map(models, tasks, data, task_suffix):

    # Set up the matplotlib figure
    plt.figure(figsize=(6, 4))

    # Create the heatmap with seaborn
    # add models and tasks to the axis

    models = [clean_model_name(model) for model in models]
    tasks = [clean_label(task) for task in tasks]
    clean_suffix = clean_task_suffix(task_suffix)

    data = pd.DataFrame(
        data,
        index=models,
        columns=tasks,
    )
    # reduce box size
    sns.set(font_scale=0.75)
    g = sns.heatmap(
        data,
        annot=True,
        cmap="coolwarm",
        vmin=0,
        vmax=500,
        linewidths=0.5,
        cbar_kws={"label": "False Positive Counts"},
        fmt="g",
    )
    g.set_xticklabels(g.get_xticklabels(), rotation=15, fontsize=8)
    g.set_yticklabels(g.get_yticklabels(), rotation=0, fontsize=8)

    # Set the plot labels and title
    plt.title(f"False Positive Leakage {clean_suffix}")

    # Show the heatmap
    plt.tight_layout()
    plt.savefig(
        f"{PRIVACY_RESULTS_DIR}/graphs/false-positive-heatmap-{task_suffix}.png"
    )
