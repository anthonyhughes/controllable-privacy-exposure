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


def plot_heat_map(models, tasks, data, task_suffix, positive_type):
     
    if positive_type == "fp":
        clean_pos_label = "False Positive"
    elif positive_type == "fn":
        clean_pos_label = "False Negatives"
        

    # Set up the matplotlib figure
    plt.figure(figsize=(4.5, 5))

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
        cmap='mako_r',
        vmin=0,
        vmax=400,
        linewidths=0.5,
        cbar_kws={"label": f"{clean_pos_label} Counts"},
        fmt="g",
        square=True,
        annot_kws={"size": 9},
    )

    g.set_xticklabels(g.get_xticklabels(), rotation=30, fontsize=8)
    g.set_yticklabels(g.get_yticklabels(), rotation=0, fontsize=8)
        
    # Adjust tick positions closer to the heatmap
    g.tick_params(axis='x', which='both', pad=-3)  # Move x-axis ticks closer
    g.tick_params(axis='y', which='both', pad=-2)  # Move y-axis ticks closer

    plt.title(f"{clean_pos_label} Leakage of Names with {clean_suffix}")

    # Show the heatmap
    plt.tight_layout()
    plt.savefig(
        f"{PRIVACY_RESULTS_DIR}/graphs/{positive_type}-heatmap-{task_suffix}.png"
    )
