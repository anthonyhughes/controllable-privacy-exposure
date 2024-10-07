import json
import matplotlib.pyplot as plt
import numpy as np
from constants import (
    FINAL_RAW_PRIVACY_RESULTS_DIR,
    PRIVACY_RESULTS_DIR,
)

def plot_grouped_precision_recall(data):
    tasks = list(data.keys())
    entities = list(data[tasks[0]].keys())  # Assuming same entities for all tasks
    
    # Prepare data for plotting
    precisions = {task: [data[task][entity]["precision"] for entity in entities] for task in tasks}
    recalls = {task: [data[task][entity]["recall"] for entity in entities] for task in tasks}
    
    x = np.arange(len(entities))  # The label locations for entities
    width = 0.2  # Width of each bar group
    
    fig, ax = plt.subplots(figsize=(12, 6))

    # Define colors and patterns for tasks
    task_colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    # Plot bars for precision for each task
    for i, task in enumerate(tasks):
        ax.bar(x + i * width - width/2, precisions[task], width, label=f'{task} Precision', 
               color=task_colors[i])

    # Plot bars for recall for each task
    for i, task in enumerate(tasks):
        ax.bar(x + (i + len(tasks)) * width - width/2, recalls[task], width, label=f'{task} Recall', 
               color=task_colors[i], alpha=0.6)

    # Add labels, title, and ticks
    ax.set_xlabel('Entity Type')
    ax.set_ylabel('Scores')
    ax.set_title('Precision and Recall Grouped by Task and Entity Type')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(entities)
    
    # Add legend
    ax.legend()

    # Function to add labels to bars
    def add_labels(rects, values):
        for rect, value in zip(rects, values):
            height = rect.get_height()
            ax.annotate(f'{value:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    # Plot labels for each bar
    for i, task in enumerate(tasks):
        precision_bars = ax.bar(x + i * width - width/2, precisions[task], width)
        recall_bars = ax.bar(x + (i + len(tasks)) * width - width/2, recalls[task], width)
        add_labels(precision_bars, precisions[task])
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
    fig.savefig(f"{PRIVACY_RESULTS_DIR}/{task_name}-all.png")


def gen_graphs(target_file):
    """
    Generate graphs for the re-identification results
    """
    with open(f"{FINAL_RAW_PRIVACY_RESULTS_DIR}/{target_file}", "r") as f:
        data = json.load(f)

    # Extract names, precision, and recall values
    # for key in data.keys():
        # task_name = key
        # plot_precision_recall(task_name, data[key])

    plot_grouped_precision_recall(data)
    print("Graphs generated successfully!")
