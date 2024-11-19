import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

from utils.graphs.graph_data import gen_data_for_ptr_utility
from utils.graphs.utils import (
    clean_label,
    clean_metric,
    clean_model_name,
    clean_property,
    fetch_clean_dataset_name,
)

mpl.rcParams["hatch.linewidth"] = 0.2

from constants import PRIVACY_RESULTS_DIR


def gen_bar_chart(true_positive_rates, tasks, file_suffix_name):
    methodologies = [
        "0-Shot Private",
        "1-Shot Private",
        "Sanitize & Summ",
    ]
    models = [
        "Claude-Sonnet-3-5",
        "GPT-4o",
        "Llama-3.1-70b-Instruct",
        "Llama-3.1-8b-Instruct",
        "Mistral-7b-Instruct",
        "(IFT) Llama-3.1-8b-Instruct",
        "(IFT) Mistral-7b-Instruct",
        "(IFT) Llama-3.1-70b-Instruct",
    ]
    patterns = ["-", "-", "|", "|", "|", "+", "+", "+"]

    # Number of methodologies and models
    n_methodologies = len(methodologies)
    n_models = len(models)

    # X locations for each group of bars
    x = np.arange(n_methodologies)

    # Bar width, ensuring that the bars fit within each methodology group
    bar_width = 0.1

    # Set up the figure
    fig, axes = plt.subplots(
        ncols=len(true_positive_rates), figsize=(16, 4), sharey=True
    )

    for idx, (true_positive_rates, ax) in enumerate(zip(true_positive_rates, axes)):
        x = np.arange(len(methodologies))
        bar_width = 0.1
        # Plot each model's bars within each methodology
        for i in range(n_models):
            # Offset for each model within a group
            offset = (i - (n_models - 1) / 2) * bar_width
            ax.bar(
                x + offset,
                [true_positive_rates[j][i] for j in range(n_methodologies)],
                width=bar_width,
                hatch=patterns[i],
                label=models[i],
            )

        # Labeling
        # ax.set_xlabel("Methodologies")
        # ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{clean_label(tasks[idx])} Summaries")
        # angling the x-axis labels
        ax.set_xticks(x)

    # Common y-axis label
    axes[0].set_ylabel("True Positive Rate")
    fig.suptitle(f"True Positive Rates of {clean_property(file_suffix_name)}")

    # Legend on the first subplot to avoid repetition
    # smaller font size in legend
    # fig legend without repearing model names
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="center",
        bbox_to_anchor=(0.5, -0.02),
        fontsize="small",
        ncol=8,
        title="Models",
    )

    plt.tight_layout()

    fig.savefig(
        f"{PRIVACY_RESULTS_DIR}/graphs/tp-rate-{file_suffix_name}.png",
        bbox_inches="tight",
    )


def gen_utility_privacy_bar_chart(
    bs_data, metric, target_task_idx, baseline_model, baseline_metrics, ax
):
    # Example Data
    n_metric = clean_metric(metric)
    models = list(bs_data.keys())
    methodologies = list(bs_data[models[0]].keys())
    model_centers = []
    current_center = 0
    gap = 0.25
    data = {}
    for model in models:
        if model not in data:
            data[model] = []
        model_data = bs_data[model]
        for methodology in methodologies:
            data[model].append(model_data[methodology][target_task_idx])

    # sort the data by the privacy score in decreasing order
    data = sorted(
        data.items(), key=lambda item: sum(t[1] for t in item[1]) / len(item[1])
    )

    data = dict(data)

    models = list(data.keys())
    methodologies = list(bs_data[models[0]].keys())

    for model in models:
        group_size = len(methodologies)  # Number of bars per group
        group_positions = np.arange(
            current_center, current_center + 0.25 * group_size, 0.25
        )
        model_centers.append(np.mean(group_positions))
        current_center += 0.25 * group_size + gap

    data = dict(data)

    # Colors for each model
    method_colors = [
        "blue",
        "orange",
        "green",
    ]

    # Preparing data for plotting
    utility_values = []
    privacy_values = []
    x_positions = []
    bar_labels = []
    colors = []
    current_x = 0  # Start x position

    for model, scores in data.items():
        for idx, (utility, privacy) in enumerate(scores):
            x_positions.append(current_x)
            utility_values.append(utility)
            privacy_values.append(privacy)
            bar_labels.append(methodologies[idx])  # Add methodology labels
            colors.append(method_colors[idx])  # Assign color to model group
            current_x += 0.25  # Small gap between bars in the same group
        current_x += gap  # Large gap between model groups

    # Convert x_positions to numpy array for plotting
    x_positions = np.array(x_positions)

    # Bar width
    bar_width = 0.25

    # Plot utility bars
    utility_bars = ax.bar(
        x_positions,
        utility_values,
        bar_width,
        label="Utility",
        color="grey",
        alpha=0.3,
    )

    # Plot privacy bars overlapping utility bars
    privacy_bars = ax.bar(
        x_positions,
        privacy_values,
        bar_width,
        label="Privacy",
        color=colors,
        alpha=0.7,
    )

    ax.set_xticks(model_centers)
    ax.set_xticklabels([clean_model_name(model) for model in models], fontsize=8)
    for i, label in enumerate(ax.get_xticklabels()):
        if i % 2 == 0:
            label.set_y(label.get_position()[1] + 0)  # Move up
        else:
            label.set_y(label.get_position()[1] - 0.025)  # Move down
    # Example of adding horizontal dashed lines at y=30 and y=70
    line_positions = [
        baseline_metrics[target_task_idx][0],
        baseline_metrics[target_task_idx][1],
    ]  # Y-coordinates for the lines
    line_colors = ["black", "black"]  # Colors for each line
    line_styles = ["--", "--"]  # Both lines are dashed
    metrics = [n_metric, "PTR"]

    for i, (y, color, style) in enumerate(
        zip(line_positions, line_colors, line_styles)
    ):
        ax.axhline(
            y=y,  # Position of the line on the y-axis
            color=color,  # Line color
            linestyle=style,  # Line style (e.g., dashed)
            linewidth=1,  # Line thickness
            alpha=0.75,  # Transparency
            label=f"Threshold {y}",  # Label for the legend
        )
        ax.text(
            x=3.5,  # X-coordinate (middle of the plot area)
            y=y,  # Y-coordinate (same as the line)
            s=f"{clean_model_name(baseline_model)} {metrics[i]} baseline",  # Text to display
            color=color,
            fontsize=8,
            ha="center",  # Horizontal alignment
            va="center",  # Vertical alignment
            backgroundcolor="white",
        )

    # Add legend
    # Create a legend for methodologies
    methodology_handles = [
        plt.Line2D(
            [0],
            [0],
            color=method_colors[0],
            marker="o",
            linestyle="",
            label="1 Shot Private",
        ),
        plt.Line2D(
            [0],
            [0],
            color=method_colors[1],
            marker="o",
            linestyle="",
            label="0 Shot Private",
        ),
        plt.Line2D(
            [0],
            [0],
            color=method_colors[2],
            marker="o",
            linestyle="",
            label="Sanitize and Summarize",
        ),
        plt.Line2D(
            [0], [0], color="grey", marker="s", linestyle="", label=f"{n_metric}"
        ),
        plt.Line2D(
            [0], [0], color=line_colors[0], linestyle="--", label=f"Utility Baseline"
        ),
        plt.Line2D(
            [0], [0], color=line_colors[1], linestyle="--", label=f"Privacy Baseline"
        ),
    ]

    # Add legend to the plot
    if target_task_idx == 0:
        ax.legend(
            handles=methodology_handles,
            title="Methodologies",
            loc="upper left",
            fontsize=9,
        )

    # Set y-axis label and title
    if target_task_idx == 0 or target_task_idx == 2:
        ax.set_ylabel("RogueL and Private Token Ratio", fontsize=10)
    dataset_label = fetch_clean_dataset_name(target_task_idx)
    ax.set_title(f"{dataset_label}", fontsize=10)

    # Add grid and adjust layout
    ax.grid(axis="y", linestyle="--", alpha=0.6)    
    return ax


def gen_utility_privacy_bar_chart_for_uber(metric):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))  # Create a new 2x2 grid
    axes = axes.flatten()  # Make it iterable

    # Generate data and plot into the new grid
    for i, ax in enumerate(axes):
        bs_data, baseline_model, baseline_metrics = gen_data_for_ptr_utility(
            utility_metric=metric
        )

        # Re-plot the chart directly into the target subplot
        gen_utility_privacy_bar_chart(
            bs_data, metric, i, baseline_model, baseline_metrics, ax=ax
        )

    fig.suptitle(f"{clean_metric(metric)} (Utility) vs. Private Token Ratio (Privacy)")
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(
        f"{PRIVACY_RESULTS_DIR}/graphs/privacy-vs-utility-uber.png",
        bbox_inches="tight",
    )
