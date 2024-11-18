import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from constants import PRIVACY_RESULTS_DIR


# Example Data
# Replace with your actual data
def gen_diff_plot_for_privacy(data):
    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Calculate the absolute difference
    df["Difference"] = df["All Positive Counts"] - df["All PII Counts"]

    # --- 1. Stacked Bar Chart ---
    plt.figure(figsize=(12, 6))
    plt.bar(
        df["Model"], df["All PII Counts"], label="(All PII Counts)", color="skyblue"
    )
    plt.bar(
        df["Model"],
        df["Difference"],
        bottom=df["All PII Counts"],
        label="True Positive Rate (Excluding All PII Counts)",
        color="orange",
    )
    plt.xlabel("Model")
    plt.ylabel("Counts")
    plt.xticks(rotation=45, ha='right')
    plt.title("Stacked Bar Chart: All PII Counts and Remaining All Positive Counts")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        f"./{PRIVACY_RESULTS_DIR}/graphs/privacy-vs-positives-1.png",
        bbox_inches="tight",
    )


    # --- 2. Absolute Difference Bar Plot ---
    # plt.figure(figsize=(12, 6))
    # plt.bar(df["Model"], df["Difference"], color="purple", alpha=0.7)
    # plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    # for i, v in enumerate(df["Difference"]):
    #     plt.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    # plt.xlabel("Model")
    # plt.ylabel("Difference (All Positive Counts - All PII Counts)")
    # plt.title("Absolute Difference Between All Positive Counts and All PII Counts")
    # plt.tight_layout()
    # plt.savefig("difference_bar_All PII Counts_All Positive Counts.png")
    # plt.savefig(
    #     f"./{PRIVACY_RESULTS_DIR}/graphs/privacy-vs-positives-2.png",
    #     bbox_inches="tight",
    # )

    # --- 3. Line Plot with Gap Highlight ---
    plt.figure(figsize=(12, 6))
    plt.plot(
        df["Model"],
        df["All Positive Counts"],
        label="True Positive Rate (All Positive Counts)",
        color="green",
        marker="o",
    )
    plt.plot(
        df["Model"],
        df["All PII Counts"],
        label="(All PII Counts)",
        color="red",
        marker="o",
    )
    plt.fill_between(
        df["Model"],
        df["All Positive Counts"],
        df["All PII Counts"],
        where=(df["All Positive Counts"] > df["All PII Counts"]),
        interpolate=True,
        color="lightgrey",
        alpha=0.5,
        label="Difference (Gap)",
    )
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Task")
    plt.ylabel("Counts")
    plt.title("Line Plot: All PII Counts vs All Positive Counts with Highlighted Gap")
    plt.legend()
    plt.tight_layout()
    # Save the figure
    plt.savefig(
        f"./{PRIVACY_RESULTS_DIR}/graphs/privacy-vs-positives-3.png",
        bbox_inches="tight",
    )
