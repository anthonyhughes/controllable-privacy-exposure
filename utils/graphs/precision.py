# def plot_precision():

#     # Data
#     methods = ["Discharge Letter", "BHC"]
#     models = ["GPT-4o", "Sonnet 3.5", "Llama-3.1-8b", "Llama-3.1-70b", "Mistral-7b"]

#     # Precision values for each model
#     # zs_priv = {
#     #     'Discharge Letter': [91, 5, 72, 90, 89],
#     #     'BHC': [0, 0, 88, 33, 93]
#     # }

#     one_s_priv = {
#         "Discharge Letter": [100, 0, 100, 93, 93],
#         "BHC": [0, 0, 100, 100, 95],
#     }

#     san_summ = {
#         "Discharge Letter": [100, 0, 100, 93, 92],
#         "BHC": [0, 0, 100, 0, 95],
#     }

#     # Plotting settings
#     x = np.arange(len(models))  # label locations
#     width = 0.2  # width of the bars
#     spacing = width  # Adjust spacing between "Discharge Letter" and "BHC" bars

#     # Creating subplots
#     fig, ax = plt.subplots(figsize=(10, 6))

#     # Plotting for Discharge Letter (shift the bars to the left)
#     # ax.bar(x - spacing, zs_priv['Discharge Letter'], width, label='ZSPriv - Discharge Letter')
#     ax.bar(
#         x - spacing + width,
#         one_s_priv["Discharge Letter"],
#         width,
#         label="OneSPriv - Discharge Letter",
#     )
#     ax.bar(
#         x - spacing + 1 * width,
#         san_summ["Discharge Letter"],
#         width,
#         label="SanSumm - Discharge Letter",
#     )

#     # Plotting for BHC (shift the bars to the right)
#     # ax.bar(x + spacing, zs_priv['BHC'], width, label='ZSPriv - BHC')
#     ax.bar(
#         x + spacing + width,
#         one_s_priv["BHC"],
#         width,
#         label="OneSPriv - BHC",
#     )
#     ax.bar(
#         x + spacing + 1 * width,
#         san_summ["BHC"],
#         width,
#         label="SanSumm - BHC",
#     )

#     # Adding labels and title
#     ax.set_xlabel("Models")
#     ax.set_ylabel("Precision (%)")
#     ax.set_title("Precision Rates of Models by Methods")
#     ax.set_xticks(x)
#     ax.set_xticklabels(models)
#     ax.legend(loc="upper left")

#     plt.tight_layout()
#     plt.show()
#     fig.savefig(f"{PRIVACY_RESULTS_DIR}/graphs/precision-all.png")