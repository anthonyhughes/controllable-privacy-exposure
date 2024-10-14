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



def plot_scatter_graph_for_util_priv():

    # Example data for sanitize and summarize - discharge letters
    models = ["GPT-4o", "Sonnet3.5", "Llama3.1-70b", "Llama3.1-8b", "Mistral7b"]
    model_complexity_set = [
        175,
        175,
        70,
        8,
        7,
    ]  # Model complexity, e.g., #parameters (in billions)
    rogue_l_scores = [12.52, 12.63, 16.18, 11.03, 12.42]
    privacy_leakage_scores = [
        0.028,
        0.011,
        0.018,
        0.022,
        0.094,
    ]  # Privacy Leakage (higher = worse)

    # Example data for ICL - discharge letters
    rogue_l_scores_2 = [17.15, 13.03, 19.47, 14.57, 14.70]
    privacy_leakage_scores_set_2 = [0.065, 0.069, 0.092, 0.035, 0.123]

    # Normalize model complexity to scale bubble sizes for visualization
    bubble_sizes_set1 = [complexity * 4 for complexity in model_complexity_set]
    bubble_sizes_set2 = [complexity * 4 for complexity in model_complexity_set]

    # Set up the figure and axis
    plt.figure(figsize=(10, 6))

    # Plot Set 1
    sc1 = plt.scatter(
        rogue_l_scores,
        privacy_leakage_scores,
        s=bubble_sizes_set1,
        alpha=0.75,
        c="blue",
        label="S&S",
        marker="o",
    )

    # Plot Set 2 (overlaid)
    sc2 = plt.scatter(
        rogue_l_scores_2,
        privacy_leakage_scores_set_2,
        s=bubble_sizes_set2,
        alpha=0.75,
        c="red",
        label="ICL",
        marker="^",
    )

    # Add labels for each model in Set 1
    for i, model in enumerate(models):
        plt.text(rogue_l_scores[i], privacy_leakage_scores[i], model, fontsize=10)

    # Add labels for each model in Set 2
    for i, model in enumerate(models):
        plt.text(
            rogue_l_scores_2[i],
            privacy_leakage_scores_set_2[i],
            model,
            fontsize=10,
        )

    # Add axis labels and title
    plt.xlabel("ROUGE-L", fontsize=12)
    plt.ylabel("Private Token Ratio (PTR)", fontsize=12)
    plt.title("Utility-Privacy Trade-off (S&S vs ICL)", fontsize=12)

    # Add grid for better readability
    plt.grid(True)

    # Add legend
    # smaller icon size in legend
    plt.legend(
        [sc1, sc2],
        ["Sanitize And Summarise (S&S)", "In-Context Learning (ICL)"],
        fontsize=10,
        loc="upper right",
    )

    # Show plot
    plt.tight_layout()
    plt.savefig(f"{PRIVACY_RESULTS_DIR}/graphs/utility-scatter-plot.png")


def plot_scatter_graph_for_sem_priv():
    # Example data for sanitize and summarize - discharge letters
    models = ["GPT-4o", "Sonnet3.5", "Llama3.1-70b", "Llama3.1-8b", "Mistral7b"]
    model_complexity_set = [
        175,
        175,
        70,
        8,
        7,
    ]  # Model complexity, e.g., #parameters (in billions)
    bert_scores = [81.66, 81.39, 81.81, 79.84, 80.36]
    privacy_leakage_scores = [
        0.028,
        0.011,
        0.018,
        0.022,
        0.094,
    ]  # Privacy Leakage (higher = worse)

    # Example data for ICL - discharge letters
    bert_scores_2 = [82.66, 81.20, 82.71, 81.59, 80.86]
    privacy_leakage_scores_set_2 = [0.065, 0.069, 0.092, 0.035, 0.123]

    # Normalize model complexity to scale bubble sizes for visualization
    bubble_sizes_set1 = [complexity * 4 for complexity in model_complexity_set]
    bubble_sizes_set2 = [complexity * 4 for complexity in model_complexity_set]

    # Set up the figure and axis
    plt.figure(figsize=(10, 6))

    # Plot Set 1
    sc1 = plt.scatter(
        bert_scores,
        privacy_leakage_scores,
        s=bubble_sizes_set1,
        alpha=0.75,
        c="blue",
        label="S&S",
        marker="o",
    )

    # Plot Set 2 (overlaid)
    sc2 = plt.scatter(
        bert_scores_2,
        privacy_leakage_scores_set_2,
        s=bubble_sizes_set2,
        alpha=0.75,
        c="red",
        label="ICL",
        marker="^",
    )

    # Add labels for each model in Set 1
    for i, model in enumerate(models):
        plt.text(bert_scores[i], privacy_leakage_scores[i], model, fontsize=10)

    # Add labels for each model in Set 2
    for i, model in enumerate(models):
        plt.text(
            bert_scores_2[i],
            privacy_leakage_scores_set_2[i],
            model,
            fontsize=10,
        )

    # Add axis labels and title
    plt.xlabel("BERTScore", fontsize=12)
    plt.ylabel("Private Token Ratio (PTR)", fontsize=12)
    plt.title("Utility-Privacy Trade-off (S&S vs ICL)", fontsize=12)

    # Add grid for better readability
    plt.grid(True)

    # Add legend
    # smaller icon size in legend
    plt.legend(
        [sc1, sc2],
        ["Sanitize And Summarise (S&S)", "In-Context Learning (ICL)"],
        fontsize=10,
        loc="upper right",
    )

    # Show plot
    plt.tight_layout()
    plt.savefig(f"{PRIVACY_RESULTS_DIR}/graphs/bert-scatter-plot.png")


def plot_precision():

    # Data
    methods = ["Discharge Letter", "BHC"]
    models = ["GPT-4o", "Sonnet 3.5", "Llama-3.1-8b", "Llama-3.1-70b", "Mistral-7b"]

    # Precision values for each model
    # zs_priv = {
    #     'Discharge Letter': [91, 5, 72, 90, 89],
    #     'BHC': [0, 0, 88, 33, 93]
    # }

    one_s_priv = {
        "Discharge Letter": [100, 0, 100, 93, 93],
        "BHC": [0, 0, 100, 100, 95],
    }

    san_summ = {
        "Discharge Letter": [100, 0, 100, 93, 92],
        "BHC": [0, 0, 100, 0, 95],
    }

    # Plotting settings
    x = np.arange(len(models))  # label locations
    width = 0.2  # width of the bars
    spacing = width  # Adjust spacing between "Discharge Letter" and "BHC" bars

    # Creating subplots
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting for Discharge Letter (shift the bars to the left)
    # ax.bar(x - spacing, zs_priv['Discharge Letter'], width, label='ZSPriv - Discharge Letter')
    ax.bar(
        x - spacing + width,
        one_s_priv["Discharge Letter"],
        width,
        label="OneSPriv - Discharge Letter",
    )
    ax.bar(
        x - spacing + 1 * width,
        san_summ["Discharge Letter"],
        width,
        label="SanSumm - Discharge Letter",
    )

    # Plotting for BHC (shift the bars to the right)
    # ax.bar(x + spacing, zs_priv['BHC'], width, label='ZSPriv - BHC')
    ax.bar(
        x + spacing + width,
        one_s_priv["BHC"],
        width,
        label="OneSPriv - BHC",
    )
    ax.bar(
        x + spacing + 1 * width,
        san_summ["BHC"],
        width,
        label="SanSumm - BHC",
    )

    # Adding labels and title
    ax.set_xlabel("Models")
    ax.set_ylabel("Precision (%)")
    ax.set_title("Precision Rates of Models by Methods")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.show()
    fig.savefig(f"{PRIVACY_RESULTS_DIR}/graphs/precision-all.png")


def plot_true_positive():

    # Data
    methods = ["Discharge Letter", "BHC"]
    models = ["GPT-4o", "Sonnet-3.5", "Llama-3.1-8b", "Llama-3.1-70b", "Mistral-7b"]

    # Precision values for each model
    # zs_priv = {
    #     'Discharge Letter': [91, 5, 72, 90, 89],
    #     'BHC': [0, 0, 88, 33, 93]
    # }

    one_s_priv = {
        "Discharge Letter": [37, 1, 13, 27, 36],
        "BHC": [0, 0, 27, 3, 40],
    }

    san_summ = {
        "Discharge Letter": [100, 0, 100, 93, 92],
        "BHC": [0, 0, 100, 0, 95],
    }

    # Plotting settings
    x = np.arange(len(models))  # label locations
    width = 0.1  # width of the bars
    spacing = width  # Adjust spacing between "Discharge Letter" and "BHC" bars

    # Creating subplots
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting for Discharge Letter (shift the bars to the left)
    # ax.bar(x - spacing, zs_priv['Discharge Letter'], width, label='ZSPriv - Discharge Letter')
    ax.bar(
        x - spacing + width,
        one_s_priv["Discharge Letter"],
        width,
        label="OneSPriv - Discharge Letter",
    )
    ax.bar(
        x - spacing + 1 * width,
        san_summ["Discharge Letter"],
        width,
        label="SanSumm - Discharge Letter",
    )

    # Plotting for BHC (shift the bars to the right)
    # ax.bar(x + spacing, zs_priv['BHC'], width, label='ZSPriv - BHC')
    ax.bar(
        x + spacing + width,
        one_s_priv["BHC"],
        width,
        label="OneSPriv - BHC",
    )
    ax.bar(
        x + spacing + 1 * width,
        san_summ["BHC"],
        width,
        label="SanSumm - BHC",
    )

    # Adding labels and title
    ax.set_xlabel("Models")
    ax.set_ylabel("Precision (%)")
    ax.set_title("Precision Rates of Models by Methods")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.show()
    fig.savefig(f"{PRIVACY_RESULTS_DIR}/graphs/precision-all.png")


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
    # Extract names, precision, and recall values
    # for key in data.keys():
    # task_name = key
    # plot_precision_recall(task_name, data[key])

    # plot_grouped_precision_recall(data)

    # plot_scatter_graph()

    # plot_precision()
    # plot_scatter_graph_for_util_priv()
    # plot_scatter_graph_for_sem_priv()
    print("Graphs generated successfully!")
