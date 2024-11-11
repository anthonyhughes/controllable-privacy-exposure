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


def plot_heat_map(models, tasks, data):

    # Set up the matplotlib figure
    plt.figure(figsize=(6, 4))

    # Create the heatmap with seaborn
    # add models and tasks to the axis

    data = pd.DataFrame(
        data,
        index=models,
        columns=tasks,
    )
    # reduce box size
    sns.set(font_scale=0.75)
    sns.heatmap(
        data,
        annot=True,
        cmap="coolwarm",
        vmin=0,
        vmax=500,
        linewidths=0.5,
        cbar_kws={"label": "True Positive Counts"},
        fmt='g',
    )

    # Set the plot labels and title
    plt.title("True Positive Leakage Across All Models and Tasks")

    # Show the heatmap
    plt.tight_layout()
    plt.savefig(f"{PRIVACY_RESULTS_DIR}/graphs/true-positive-heatmap.png")

