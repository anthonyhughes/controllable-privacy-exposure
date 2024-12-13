import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

from utils.graphs.bar_charts_v2 import gen_utility_privacy_bar_chart
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


def gen_combined_utility_privacy_bar_chart(metric):
    # Create a 3x4 subplot grid
    fig, axes = plt.subplots(3, 4, figsize=(24, 15))

    # Flatten axes to make it easier to iterate
    axes = axes.flatten()

    # First 2 rows: data from gen_data_for_ptr_utility
    tpr_data = gen_data_for_tpr_utility(utility_metric=metric)
    ptr_data, _, _ = gen_data_for_ptr_utility(
            utility_metric=metric
        )

    for i in range(4):
        gen_utility_privacy_bar_chart(
            ptr_data,
            tpr_data,
            metric,
            i,
            "",
            [],
            axes=axes,
            util_ylim=(0, 0.28),
            priv_ylim=(0, 0.65)
        )

    # # Add a shared title for the figure
    # fig.suptitle(
    #     f"Utility ({clean_metric(metric)}) vs. Privacy/Exposure Metrics",
    #     fontsize=18,
    #     y=1.02,
    # )

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(
        f"{PRIVACY_RESULTS_DIR}/graphs/combined-privacy-utility-{metric}.png",
        bbox_inches="tight",
        dpi=1200,
    )
