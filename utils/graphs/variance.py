import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from constants import PRIVACY_RESULTS_DIR


def gen_variance_graph(data, file_suffix):
    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Find the variance with the lowest PTR
    lowest_ptr = df.loc[df['PTR'].idxmin()]

    # Bar chart visualization
    plt.figure(figsize=(10, 8))
    bars = plt.bar(df["Variant"], df["PTR"], color="skyblue", edgecolor="black")

    # Highlight the bar with the lowest PTR
    for bar, ptr in zip(bars, df["PTR"]):
        if ptr == lowest_ptr["PTR"]:
            bar.set_color("orange")
            bar.set_edgecolor("red")

    # Add annotations
    for i, ptr in enumerate(df["PTR"]):
        plt.text(i, ptr + 0.005, f"{ptr:.2f}", ha='center', fontsize=10)

    # Labels and title
    plt.xlabel("Prompt Prefix Variant", fontsize=12)
    plt.ylabel("Private Token Ratio (PTR)", fontsize=12)
    plt.title("Avg. PTR per Prompt Variant", fontsize=14)
    plt.tight_layout()

    # Save the figure
    plt.savefig(
        f"./{PRIVACY_RESULTS_DIR}/graphs/privacy-variance-{file_suffix}.png",
        bbox_inches="tight",
    )


def gen_std_variance_graph(data, file_suffix):
    # Convert to arrays
    variances = data["Variant"]
    means = data["PTR_Mean"]
    std_devs = data["PTR_Std"]

    # Create bar chart with error bars
    plt.figure(figsize=(10, 6))
    bars = plt.bar(variances, means, yerr=std_devs, capsize=5, color="skyblue", edgecolor="black", alpha=0.8)

    # Highlight the bar with the lowest mean PTR
    min_mean_index = np.argmin(means)
    bars[min_mean_index].set_color("orange")
    bars[min_mean_index].set_edgecolor("red")

    # Add annotations
    for i, (mean, std) in enumerate(zip(means, std_devs)):
        plt.text(i, mean + 0.005, f"{mean:.2f} ± {std:.2f}", ha="center", fontsize=10)

    # Labels and title
    plt.xlabel("Prompt Prefix Variant", fontsize=12)
    plt.ylabel("Private Token Ratio (PTR)", fontsize=12)
    plt.title("Prompt Prefix Variant Impact on PTR", fontsize=14)
    plt.tight_layout()

        # Save the figure
    plt.savefig(
        f"./{PRIVACY_RESULTS_DIR}/graphs/privacy-mean-std-variance-{file_suffix}.png",
        bbox_inches="tight",
    )