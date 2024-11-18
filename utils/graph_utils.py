from utils.graphs.bar_chart import gen_bar_chart
from utils.graphs.graph_data import (
    gen_data_for_all_properties_comparison,
    gen_data_for_property_comparison,
    gen_data_for_ptr_utility,
    gen_data_for_ptr_variation,
    gen_data_for_ptr_mean_std_variation,
    gen_data_for_document_length,
    gen_false_positives_for_heat_map,
    gen_ptr_tp_data
)
from utils.graphs.heatmaps import plot_heat_map
from utils.graphs.scatters import plot_ptr_vs_tpr
from utils.graphs.simple import gen_utility_privacy_graph
from utils.graphs.variance import gen_std_variance_graph, gen_variance_graph
from utils.graphs.diff_plot import gen_diff_plot_for_privacy


def gen_graphs():
    """
    Generate graphs for the re-identification results
    """

    # heat_datasets, target_tasks = gen_data_for_all_properties_comparison()
    # gen_bar_chart(heat_datasets, target_tasks, file_suffix_name="all_properties")

    # heat_datasets, target_tasks = gen_data_for_property_comparison(property_name="PERSON")
    # gen_bar_chart(heat_datasets, target_tasks, file_suffix_name="names")

    # heat_datasets, target_tasks = gen_data_for_property_comparison(property_name="DATE")
    # gen_bar_chart(heat_datasets, target_tasks, file_suffix_name="dates")

    # heat_datasets, target_tasks = gen_data_for_property_comparison(property_name="ORG")
    # gen_bar_chart(heat_datasets, target_tasks, file_suffix_name="org")

    # metric = "bertscore"
    # bs_data = gen_data_for_ptr_utility(utility_metric=metric)
    # gen_utility_privacy_graph(bs_data, metric, metric_range=(0.76, 0.875))

    # metric = "rougeL"
    # bs_data = gen_data_for_ptr_utility(utility_metric=metric)
    # gen_utility_privacy_graph(bs_data, metric, metric_range=(0.05, 0.3))

    # metric = "bertscore"
    # bs_data = gen_data_for_ptr_utility(utility_metric=metric, privacy_metric="pii_document_percentage")
    # gen_utility_privacy_graph(bs_data, metric, metric_range=(0.76, 0.875), privacy_range=(0, 1), privacy_metric="pii_document_percentage")

    # bs_data = gen_data_for_ptr_variation(privacy_metric="private_token_ratio")
    # gen_variance_graph(bs_data, file_suffix="private_token_ratio")

    # bs_data = gen_data_for_ptr_mean_std_variation(privacy_metric="private_token_ratio")
    # gen_std_variance_graph(bs_data, file_suffix="private_token_ratio")

    # bs_data = gen_data_for_document_length(privacy_metric="private_token_ratio")
    
    # bs_data = gen_false_positives_for_heat_map(task_suffix="")
    # plot_heat_map(bs_data[0], bs_data[1], bs_data[2], task_suffix="")

    # bs_data = gen_false_positives_for_heat_map(task_suffix="_in_context")
    # plot_heat_map(bs_data[0], bs_data[1], bs_data[2], task_suffix="_in_context")

    # bs_data = gen_false_positives_for_heat_map(task_suffix="_sani_summ")
    # plot_heat_map(bs_data[0], bs_data[1], bs_data[2], task_suffix="_sani_summ")

    bs_data = gen_ptr_tp_data()
    # plot_ptr_vs_tpr(bs_data)
    gen_diff_plot_for_privacy(bs_data)
    