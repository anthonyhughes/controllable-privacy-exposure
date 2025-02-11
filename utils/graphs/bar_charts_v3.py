import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


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


import matplotlib.gridspec as gridspec

def gen_combined_utility_privacy_bar_chart():
    fig = plt.figure(figsize=(24, 15))
    gs = gridspec.GridSpec(5, 3, height_ratios=[0.5, 0.4, 0.4, 0.4, 0.4])

    # Create axes for all subplots
    axes = [fig.add_subplot(gs[i, j]) for i in range(5) for j in range(3)]

    # First 2 rows: data from gen_data_for_ptr_utility
    tpr_data = gen_data_for_tpr_utility(utility_metric="bertscore")
    ptr_data, _, _ = gen_data_for_ptr_utility(utility_metric="rougeL")

    for i in range(3):
        gen_utility_privacy_bar_chart(
            ptr_data,
            tpr_data,
            "roguel",
            i,
            "",
            [],
            axes=axes,
            util_ylim=(0.05, 0.275),
            util2_ylim=(0.75, 0.9),
            priv_ylim=(0, 0.65),
            expo_ylim=(0, 0.6)
        )

    # Adjust layout and save
    plt.tight_layout()
    pp = PdfPages(f"{PRIVACY_RESULTS_DIR}/graphs/combined-privacy-utility-compressed.pdf")
    plt.savefig(
        f"{PRIVACY_RESULTS_DIR}/graphs/combined-privacy-utility-compressed.png",
        bbox_inches="tight",
        dpi=300,
    )
    pp.savefig(fig)
    pp.close()

