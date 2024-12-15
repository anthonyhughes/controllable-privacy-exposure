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
        f"{PRIVACY_RESULTS_DIR}/graphs/{positive_type}-heatmap-{task_suffix}.png",
        dpi=1200
    )

def plot_heat_maps_side_by_side(models, tasks, data_fp, data_fn, task_suffix):
    # Set up the matplotlib figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))  # 1 row, 2 columns

    # Clean model and task labels
    models = [clean_model_name(model) for model in models]
    tasks = [clean_label(task) for task in tasks]

    # Create a shared color bar scale
    vmin, vmax = 0, 1000

    # Data for False Positives heatmap
    data_fp = pd.DataFrame(data_fp, index=models, columns=tasks)
    # sns.set(font_scale=0.75)
    fp_plot = sns.heatmap(
        data_fp,
        annot=True,
        cmap='mako_r',
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        fmt="g",
        square=True,
        annot_kws={"size": 8},
        ax=axes[0],
        cbar=False  # Disable individual color bar
    )
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=25, ha='right', rotation_mode='anchor', va='top', fontsize=9)
    axes[0].tick_params(axis='y', labelsize=9)

    # Data for False Negatives heatmap
    data_fn = pd.DataFrame(data_fn, index=models, columns=tasks)
    fn_plot = sns.heatmap(
        data_fn,
        annot=True,
        cmap='mako_r',
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        fmt="g",
        square=True,
        annot_kws={"size": 8},
        ax=axes[1],
        cbar=False  # Disable individual color bar
    )
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=25, ha='right', rotation_mode='anchor', va='top', fontsize=9)
    axes[1].tick_params(axis='y', left=False)  # Disable y-axis ticks and labels
    axes[1].set_yticks([])  # Remove y-axis labels for the second heatmap

    # Add a shared color bar to the right of the second heatmap
    cbar = fig.colorbar(
        fn_plot.collections[0],  # Link color bar to the second heatmap
        ax=axes,  # Apply to both axes
        location="right",
        fraction=0.03,  # Width of the color bar
        pad=0  # Padding between the color bar and the heatmap
    )
    # Customize the color bar ticks
    ticks = cbar.get_ticks()  # Get current ticks
    ticks[-1] = vmax  # Ensure the last tick matches the maximum value
    tick_labels = [f"{tick:.0f}" for tick in ticks[:-1]] + [">1000"]  # Replace the last label
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(tick_labels)
    cbar.ax.set_position([cbar.ax.get_position().x0 - 0.13,  # Align left with second heatmap
                        axes[0].get_position().y0 + 0,  # Align top with first heatmap
                        cbar.ax.get_position().width, 
                        axes[0].get_position().height]) 
    # cbar.set_label("False Positive/Negative Rates", fontsize=9)

    # Adjust layout to avoid overlapping
    fig.subplots_adjust(right=0.71, wspace=-0.4)  # Ensure space for the color bar

    # plt.tight_layout()
    # Save the figure
    plt.savefig(
        f"{PRIVACY_RESULTS_DIR}/graphs/fp-fn-heatmaps-{task_suffix}.svg",
        dpi=300,
        bbox_inches='tight'
    )
