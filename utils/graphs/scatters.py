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
from scipy.stats import linregress

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

def plot_ptr_vs_tpr(ptr_data):    
    # Convert to DataFrame
    df = pd.DataFrame(ptr_data)

    # Create subplots for TPR and FPR vs PTR
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Function to add regression line and display slope
    def plot_relationship(ax, x, y, xlabel, ylabel, title):
        sns.scatterplot(x=x, y=y, hue="Model", data=df, ax=ax, palette="Set1", s=100)
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        sns.lineplot(
            x=x, y=slope * x + intercept, ax=ax, color="black", label=f"Trend (R² = {r_value**2:.2f})"
        )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()

    # Plot TPR vs PTR
    plot_relationship(
        axes[0], df["All PII Counts"], df["All Positive Counts"], "Private Token Counts", "True and False Positive Rate", "Private vs Identifiable"
    )

    # # Plot FPR vs PTR
    # plot_relationship(
    #     axes[1], df["PTR"], df["False Positive Rate"], "Private Token Ratio (PTR)", "False Positive Rate", "FPR vs PTR"
    # )

    # Adjust layout
    plt.tight_layout()
    plt.suptitle("Relationship Between Privacy and Identifiability", y=1)
    plt.tight_layout()
    plt.savefig(
        f"{PRIVACY_RESULTS_DIR}/graphs/tpr-ptr-scatter.png"
    )
