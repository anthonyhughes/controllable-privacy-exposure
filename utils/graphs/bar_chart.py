import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from utils.graphs.utils import clean_label

mpl.rcParams["hatch.linewidth"] = 0.2

from constants import PRIVACY_RESULTS_DIR


def gen_bar_chart(true_positive_rates, tasks, file_suffix_name):
    # Sample data for demonstration
    # true_positive_rates = [
    #     [0.75, 0.80, 0.78, 0.82, 0.85, 0.87, 0.83],  # Methodology 1
    #     [0.68, 0.74, 0.72, 0.70, 0.76, 0.79, 0.75],  # Methodology 2
    #     [0.85, 0.88, 0.84, 0.89, 0.90, 0.91, 0.87],  # Methodology 3
    #     [0.72, 0.75, 0.73, 0.78, 0.81, 0.82, 0.80],  # Methodology 4
    # ]

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
    fig, axes = plt.subplots(ncols=len(true_positive_rates), figsize=(16, 4), sharey=True)

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
        ax.set_title(f"TPR for {clean_label(tasks[idx])} Task")
        # angling the x-axis labels        
        ax.set_xticks(x)
        ax.set_xticklabels(methodologies, rotation=20)
    
    # Common y-axis label
    axes[0].set_ylabel("True Positive Rate")
    fig.suptitle(f"True Positive Rates of {file_suffix_name.title()}")

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

    fig.savefig(f"{PRIVACY_RESULTS_DIR}/graphs/tp-rate-{file_suffix_name}.png", bbox_inches="tight")
