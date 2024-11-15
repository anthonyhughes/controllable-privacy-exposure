from utils.graphs.bar_chart import gen_bar_chart
from utils.graphs.graph_data import (
    gen_data_for_all_properties_comparison,
    gen_data_for_property_comparison,
    gen_data_for_ptr_utility,
)
from utils.graphs.simple import gen_utility_privacy_graph


def gen_graphs():
    """
    Generate graphs for the re-identification results
    """

    heat_datasets, target_tasks = gen_data_for_all_properties_comparison()
    gen_bar_chart(heat_datasets, target_tasks, file_suffix_name="all_properties")

    heat_datasets, target_tasks = gen_data_for_property_comparison(property_name="PERSON")
    gen_bar_chart(heat_datasets, target_tasks, file_suffix_name="names")

    heat_datasets, target_tasks = gen_data_for_property_comparison(property_name="DATE")
    gen_bar_chart(heat_datasets, target_tasks, file_suffix_name="dates")

    heat_datasets, target_tasks = gen_data_for_property_comparison(property_name="ORG")
    gen_bar_chart(heat_datasets, target_tasks, file_suffix_name="org")

    metric = "bertscore"
    bs_data = gen_data_for_ptr_utility(utility_metric=metric)
    gen_utility_privacy_graph(bs_data, metric, metric_range=(0.76, 0.875))

    metric = "rougeL"
    bs_data = gen_data_for_ptr_utility(utility_metric=metric)
    gen_utility_privacy_graph(bs_data, metric, metric_range=(0.05, 0.3))

    # metric = "bertscore"
    # bs_data = gen_data_for_ptr_utility(utility_metric=metric, privacy_metric="pii_document_percentage")
    # gen_utility_privacy_graph(bs_data, metric, metric_range=(0.76, 0.875), privacy_range=(0, 1), privacy_metric="pii_document_percentage")

