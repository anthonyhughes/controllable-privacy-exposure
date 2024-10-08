import json
import matplotlib.pyplot as plt
import numpy as np
from constants import (
    FINAL_RAW_PRIVACY_RESULTS_DIR,
    FINAL_REID_RESULTS_DIR,
    PRIVACY_RESULTS_DIR,
)
import seaborn as sns


def plot_grouped_precision_recall(data):
    tasks = list(data.keys())
    entities = list(data[tasks[0]].keys())  # Assuming same entities for all tasks

    # Prepare data for plotting
    # precisions = {task: [data[task][entity]["precision"] for entity in entities] for task in tasks}
    recalls = {
        task: [data[task][entity]["recall"] for entity in entities] for task in tasks
    }

    x = np.arange(len(entities))  # The label locations for entities
    width = 0.2  # Width of each bar group

    fig, ax = plt.subplots(figsize=(6, 6))

    # Define colors and patterns for tasks
    task_colors = [
        "tab:blue",
        "tab:orange",
        "tab:red",
        "tab:green",
        "tab:purple",
        "tab:brown",
    ]

    # Plot bars for precision for each task
    # for i, task in enumerate(tasks):
    #     ax.bar(x + i * width - width/2, precisions[task], width, label=f'{task} Precision',
    #            color=task_colors[i])

    # Plot bars for recall for each task
    for i, task in enumerate(tasks):
        ax.bar(
            x + (i + len(tasks)) * width - width / 2,
            recalls[task],
            width,
            label=f"{task} TPR",
            color=task_colors[i],
            alpha=0.6,
        )

    # Add labels, title, and ticks
    ax.set_xlabel("Entity Type")
    ax.set_ylabel("Scores")
    ax.set_title("TPR Grouped by Task and Entity Type")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(entities)

    # Add legend
    ax.legend()

    # Function to add labels to bars
    def add_labels(rects, values):
        for rect, value in zip(rects, values):
            height = rect.get_height()
            ax.annotate(
                f"{value:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    # Plot labels for each bar
    for i, task in enumerate(tasks):
        recall_bars = ax.bar(
            x + (i + len(tasks)) * width - width / 2, recalls[task], width
        )
        add_labels(recall_bars, recalls[task])

    plt.tight_layout()
    plt.show()


# Function to plot Precision and Recall for each task
def plot_precision_recall(task_name, entity_data):
    entities = list(entity_data.keys())
    precisions = [entity_data[entity]["precision"] for entity in entities]
    recalls = [entity_data[entity]["recall"] for entity in entities]

    x = np.arange(len(entities))  # Label locations
    width = 0.35  # Width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width / 2, precisions, width, label="Precision")
    rects2 = ax.bar(x + width / 2, recalls, width, label="Recall")

    # Add some text for labels, title, and custom x-axis tick labels
    ax.set_xlabel("Entity Type")
    ax.set_ylabel("Scores")
    ax.set_title(f"Precision and Recall for Task: {task_name}")
    ax.set_xticks(x)
    ax.set_xticklabels(entities)
    ax.legend()

    # Add values on top of the bars
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    add_labels(rects1)
    add_labels(rects2)

    plt.tight_layout()
    plt.show()
    fig.savefig(f"{PRIVACY_RESULTS_DIR}/graphs/{task_name}-all.png")


def plot_heat_map(data):
    # Convert dictionary into a DataFrame for easy plotting
    import pandas as pd

    df = pd.DataFrame(data).T  # Transpose for correct orientation (models as rows)

    # Set up the matplotlib figure
    plt.figure(figsize=(10, 6))

    # Create the heatmap with seaborn
    sns.heatmap(
        df,
        annot=True,
        cmap="coolwarm",
        linewidths=0.5,
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Leakage Score"},
    )

    # Set the plot labels and title
    plt.title("Information Leakage Across Language Models", fontsize=16)
    plt.ylabel("Language Models")
    plt.xlabel("Information Types")

    # Show the heatmap
    plt.tight_layout()
    plt.show()


def plot_scatter_graph():

    # Example data for discharge letters
    models = ['Sonnet3.5', 'GPT-4o', 'Llama3.1-70b', 'Llama3.1-8b', 'Mistral7b']  
    utility_scores = [0.23, 0.23, 0.25, 0.18, 0.22]
    privacy_leakage_scores = [0.011, 0.028, 0.018, 0.002, 0.094]  # Privacy Leakage (higher = worse)
    model_complexity_set = [175, 175, 70, 8, 7]  # Model complexity, e.g., #parameters (in billions)

    # Second set of results (Set 2)
    models_set_2 = ['Sonnet3.5', 'GPT-4o', 'Llama3.1-70b', 'Llama3.1-8b', 'Mistral7b']  
    utility_scores_set_2 = [0.27, 0.27, 0.22, 0.17, 0.26]
    privacy_leakage_scores_set_2 = [0.022, 0.013, 0.031, 0.014, 0.120]

    # Normalize model complexity to scale bubble sizes for visualization
    bubble_sizes_set1 = [complexity * 8 for complexity in model_complexity_set]
    bubble_sizes_set2 = [complexity * 8 for complexity in model_complexity_set]

    # Set up the figure and axis
    plt.figure(figsize=(10, 6))

    # Plot Set 1
    sc1 = plt.scatter(utility_scores, privacy_leakage_scores, 
                    s=bubble_sizes_set1, alpha=0.5, c='blue', label='Discharge', marker='o')

    # Plot Set 2 (overlaid)
    sc2 = plt.scatter(utility_scores_set_2, privacy_leakage_scores_set_2, 
                    s=bubble_sizes_set2, alpha=0.5, c='red', label='Hospital Course', marker='^')

    # Add labels for each model in Set 1
    for i, model in enumerate(models):
        plt.text(utility_scores[i] - 0.01, privacy_leakage_scores[i], model, fontsize=9)

    # Add labels for each model in Set 2
    for i, model in enumerate(models_set_2):
        plt.text(utility_scores_set_2[i] - 0.01, privacy_leakage_scores_set_2[i], model, fontsize=9)

    # Add axis labels and title
    plt.xlabel('Utility Score ROUGE-1', fontsize=12)
    plt.ylabel('Private Token Ratio', fontsize=12)
    plt.title('Trade-off between Utility and Privacy Leakage (DI vs BHC)', fontsize=14)

    # Add grid for better readability
    plt.grid(True)

    # Add legend
    # smaller icon size in legend
    plt.legend([sc1, sc2], ['Discharge', 'Hospital Course'], fontsize=10, loc='upper left')
    # plt.legend(bbox_to_anchor=(1, 1), fontsize=10)

    # Show plot
    plt.tight_layout()
    plt.show()


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

    # heat_data = {}
    # for file in files:
    #     with open(f"{FINAL_REID_RESULTS_DIR}/{file}", "r") as f:            
    #         model = file.split("reidentification")[0][:-1]
    #         data = json.load(f)
    #         if model not in heat_data:
    #             data[model] = {}
    #         true_positives = data["true_positives"]
            

    # Extract names, precision, and recall values
    # for key in data.keys():
    # task_name = key
    # plot_precision_recall(task_name, data[key])

    # plot_grouped_precision_recall(data)
    # plot_heat_map(data)

    plot_scatter_graph()
    print("Graphs generated successfully!")
