import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

from utils.graphs.graph_data import gen_data_for_ptr_utility, gen_data_for_tpr_utility
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
    bs_data,
    metric,
    target_task_idx,
    baseline_model,
    baseline_metrics,
    axes,
    privacy_metric="tpr",
):
    # Process data
    n_metric = clean_metric(metric)
    models = list(bs_data.keys())
    methodologies = list(bs_data[models[0]].keys())

    # Prepare data structure
    data = {}
    for model in models:
        if model not in data:
            data[model] = []
        model_data = bs_data[model]
        for methodology in methodologies:
            data[model].append(model_data[methodology][target_task_idx])

    # Sort data by average score
    # data = dict(
    #     sorted(data.items(), key=lambda item: sum(t[1] for t in item[1]) / len(item[1]))
    # )

    models = list(data.keys())
    method_colors = ["orange", "blue", "green"]

    # Calculate bar positions
    gap = 0.25
    model_centers = []
    current_center = 0
    for model in models:
        group_size = len(methodologies)
        group_positions = np.arange(
            current_center, current_center + 0.25 * group_size, 0.25
        )
        model_centers.append(np.mean(group_positions))
        current_center += 0.25 * group_size + gap

    # Prepare plotting data
    x_positions = []
    utility_values = []
    privacy_values = []
    colors = []
    current_x = 0

    for model, scores in data.items():
        for idx, (utility, privacy) in enumerate(scores):
            x_positions.append(current_x)
            utility_values.append(utility)
            privacy_values.append(privacy)
            colors.append(method_colors[idx])
            current_x += 0.25
        current_x += gap

    x_positions = np.array(x_positions)
    bar_width = 0.25

    # Plot utility bars (top row)
    ax_utility = axes[target_task_idx]
    utility_bars = ax_utility.bar(
        x_positions, utility_values, bar_width, color=colors, alpha=0.7
    )

    # Plot privacy bars (bottom row)
    ax_privacy = axes[target_task_idx + 4]
    privacy_bars = ax_privacy.bar(
        x_positions, privacy_values, bar_width, color=colors, alpha=0.7
    )

    # Configure both plots
    for i, (ax, values, metric_name) in enumerate([
        (ax_utility, utility_values, n_metric),
        (ax_privacy, privacy_values, privacy_metric.upper()),
    ]):
        ax.set_xticks(model_centers)
        ax.set_xticklabels(
            [clean_model_name(model) for model in models], fontsize=8, rotation=45
        )

        if baseline_model:
            baseline_value = baseline_metrics[target_task_idx][
                0 if metric_name == n_metric else 1
            ]
            ax.axhline(
                y=baseline_value,
                color="black",
                linestyle="--",
                linewidth=1,
                alpha=0.75,
                label=f"Baseline {baseline_value:.2f}",
            )
            ax.text(
                x=max(x_positions),
                y=baseline_value,
                s=f"{clean_model_name(baseline_model)} Baseline",
                color="black",
                fontsize=8,
                ha="right",
                va="bottom",
                backgroundcolor="white",
            )

        ax.grid(axis="y", linestyle="--", alpha=0.6)
        dataset_label = fetch_clean_dataset_name(target_task_idx)
        ax.set_title(f"{dataset_label}", fontsize=10)

    # Add legend to first column only
    if target_task_idx == 0:
        methodology_handles = [
            plt.Line2D([0], [0], color=color, marker="o", linestyle="", label=label)
            for color, label in zip(
                method_colors,
                ["0 Shot Private", "1 Shot Private", "Sanitize and Summarize"],
            )
        ]
        if baseline_model:
            methodology_handles.append(
                plt.Line2D([0], [0], color="black", linestyle="--", label="Baseline")
            )
        ax_utility.legend(
            handles=methodology_handles,
            title="Methodologies",
            loc="upper left",
            fontsize=9,
        )
    # set y axis label
    if target_task_idx == 0:
        ax_utility.set_ylabel(n_metric)            
        ax_privacy.set_ylabel(clean_metric(privacy_metric))


def gen_utility_privacy_bar_chart_for_uber(metric, privacy_metric):
    # Create 2x4 subplot grid
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()  # Make it easier to iterate

    # Generate data and plot into the new grid
    for i in range(4):  # For each task
        bs_data, baseline_model, baseline_metrics = gen_data_for_ptr_utility(
            utility_metric=metric
        )
        gen_utility_privacy_bar_chart(
            bs_data,
            metric,
            i,
            "",
            [],
            axes=axes,
            privacy_metric=privacy_metric,
        )

    fig.suptitle(
        f"Utility ({clean_metric(metric)}) vs. Privacy ({clean_metric(privacy_metric)})",
        fontsize=14,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(
        f"{PRIVACY_RESULTS_DIR}/graphs/privacy-vs-utility-{metric}-vs-{privacy_metric}-separated.png",
        bbox_inches="tight",
        dpi=300,
    )


def gen_utility_privacy_bar_chart_for_tpr_uber(metric, privacy_metric):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # Create a new 2x2 grid
    axes = axes.flatten()  # Make it iterable

    for i in range(4):  # For each task
        bs_data = gen_data_for_tpr_utility(utility_metric=metric)
        gen_utility_privacy_bar_chart(
            bs_data,
            metric,
            i,
            "",
            [],
            axes=axes,
            privacy_metric=privacy_metric,
        )

    fig.suptitle(f"Utility ({clean_metric(metric)}) vs. Exposure (True Positive Rate)")
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(
        f"{PRIVACY_RESULTS_DIR}/graphs/exposure-vs-utility-seperated.png",
        bbox_inches="tight",
        dpi=300,
    )
