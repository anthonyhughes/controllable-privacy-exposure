import matplotlib.pyplot as plt

from constants import EVAL_MODELS_REAL_MAPPING, PRIVACY_RESULTS_DIR
from utils.graphs.utils import clean_label, clean_metric, clean_model_name, clean_privacy_metric


# Utility function to find Pareto-efficient points with the lowest privacy and highest utility
def pareto_front(points):
    # Sort points by utility (descending) and then by privacy (ascending)
    sorted_points = sorted(points, key=lambda x: (-x[0], x[1]))

    # Start with the first point in the sorted list
    pareto_points = [sorted_points[0]]

    # Iterate over sorted points, selecting only those with lower privacy for the same utility level
    for point in sorted_points[1:]:
        # Only add the point if it has a lower privacy score for a given or higher utility
        if point[0] == pareto_points[-1][0] and point[1] < pareto_points[-1][1]:
            pareto_points[-1] = point  # Replace with the point having lower privacy
        elif point[0] < pareto_points[-1][0]:
            pareto_points.append(point)

    return pareto_points


def gen_utility_privacy_graph(
    data, metric, metric_range=(0.75, 0.875), privacy_range=(0, 0.6), privacy_metric="private_token_ratio"
):
    # Set up the figure with a 4x2 grid of subplots
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
    axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

    # Colors for each methodology line
    methodology_colors = {
        "0 Shot": "blue",
        "1 Shot": "green",
        "Sanitize and Summarize": "orange",
    }

    methodology_markers = {
        "0 Shot": "o",
        "1 Shot": "+",
        "Sanitize and Summarize": "v",
    }

    dataset_colors = {
        "brief_hospital_course": "blue",
        "cnn": "green",
        "discharge_instructions": "orange",
        "legal_court": "red",
    }

    # Markers for each dataset
    data_set_marks = {
        "brief_hospital_course": "o",
        "cnn": "v",
        "discharge_instructions": "s",
        "legal_court": "p",
    }

    # Markers for each dataset
    data_set_markers = [
        "o",
        "v",
        "s",
        "p",
    ]

    # Loop through each model and corresponding subplot
    for idx, (model_name, methodologies) in enumerate(data.items()):
        ax = axes[idx]
        dataset_points = {}
        for methodology, points_by_dataset in methodologies.items():
            # color = methodology_colors[methodology]  # Get color for methodology

            for i, points in enumerate(points_by_dataset):
                dataset_name = list(data_set_marks.keys())[i]  # Get dataset name
                dataset_name_clean = clean_label(dataset_name)  # Clean dataset name
                # marker = data_set_markers[i]  # Get marker for dataset
                utility_scores, privacy_scores = points

                # Plot each methodology and dataset combination with its color and marker
                ax.scatter(
                    utility_scores,
                    privacy_scores,
                    label=f"{methodology}",
                    color=dataset_colors[dataset_name],
                    marker=methodology_markers[methodology],
                    alpha=0.7,
                )
                # Group dataset points for line drawing
                if dataset_name not in dataset_points:
                    dataset_points[dataset_name] = []
                dataset_points[dataset_name].append((utility_scores, privacy_scores))

        # Set title and labels for each subplot
        model_name = clean_model_name(model_name)
        ax.set_title(model_name)
        ax.set_xlim(metric_range[0], metric_range[1])
        ax.set_ylim(privacy_range[0], privacy_range[1])
        ax.grid(axis="both", linestyle="--", )

        # # Draw lines connecting methodologies for the same dataset
        # for dataset_name, points in dataset_points.items():
        #     points = sorted(points, key=lambda x: x[0])  # Sort points by utility score
        #     utility_scores, privacy_scores = zip(*points)
        #     ax.plot(
        #         utility_scores,
        #         privacy_scores,
        #         label=f"{dataset_name} (connection)",
        #         color=dataset_colors[dataset_name],
        #         linestyle="--",
        #         alpha=0.6,
        #     )            

    axes[0].set_ylabel("Private Token Ratio")
    axes[4].set_ylabel("Private Token Ratio")
    n_metric = clean_metric(metric)
    n_privacy_metric = clean_privacy_metric(privacy_metric)
    axes[4].set_xlabel(f"{n_metric}")
    axes[5].set_xlabel(f"{n_metric}")
    axes[6].set_xlabel(f"{n_metric}")
    axes[7].set_xlabel(f"{n_metric}")

    # Adjust layout to prevent overlap and make space for the title
    plt.tight_layout()
    fig.suptitle(
        f"Utility ({clean_metric(n_metric)}) vs. Privacy ({n_privacy_metric})", y=1.01
    )

        # Create legend for methodologies
    methodology_handles = [
        plt.Line2D([0], [0], color=color, marker=methodology_markers[method], linestyle='', label=method)
        for method, color in methodology_colors.items()
    ]
    fig.legend(
        handles=methodology_handles,
        loc="lower left",
        ncol=3,
        markerscale=1.5,
        title="Methodologies",
        bbox_to_anchor=(0.1, -0.15),  # Adjust position
        prop={"size": 10},
    )

    # Create legend for datasets
    dataset_handles = [
        plt.Line2D([0], [0], color=colour, linestyle='--', label=dataset)
        for colour, (dataset, marker) in zip(dataset_colors.values(), data_set_marks.items())
    ]
    fig.legend(
        handles=dataset_handles,
        loc="lower right",
        ncol=4,
        markerscale=1.5,
        title="Datasets",
        bbox_to_anchor=(0.9, -0.15),  # Adjust position
        prop={"size": 10},
    )


    # Save the figure
    fig.savefig(
        f"./{PRIVACY_RESULTS_DIR}/graphs/utility-privacy-{metric}-{privacy_metric}.png",
        bbox_inches="tight",
    )
