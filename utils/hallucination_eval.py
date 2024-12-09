from summac.model_summac import SummaCZS, SummaCConv
from evaluate import load


def run_hallucination_eval(target_model):
    print(f"Running hallucination evaluation for model: {target_model}")
    print("Loading evals...")

    all_resulits = {
        "harim_plus": [],
        "summaczs": [],
        "summacconv": [],
    }

    
    document = """Scientists are studying Mars to learn about the Red Planet and find landing sites for future missions.
        One possible site, known as Arcadia Planitia, is covered instrange sinuous features.
        The shapes could be signs that the area is actually made of glaciers, which are large masses of slow-moving ice.
        Arcadia Planitia is in Mars' northern lowlands.
        """
    summary1 = "There are strange shape patterns on Arcadia Planitia. The shapes could indicate the area might be made of glaciers. This makes Arcadia Planitia ideal for future missions."
    summary2 = "There are strange shape patterns on Arcadia Planitia. The shapes could indicate the area might be made of glaciers."

    # Hallucination Risk Measurement
    harim_plus = load("NCSOFT/harim_plus")

    model_zs = SummaCZS(
        granularity="sentence", model_name="vitc", device="cpu"
    )  # If you have a GPU: switch to: device="cuda"
    model_conv = SummaCConv(
        models=["vitc"],
        bins="percentile",
        granularity="sentence",
        nli_labels="e",
        device="cpu",
        start_file="default",
        agg="mean",
    )

    harim_result = harim_plus.compute(
        predictions=[summary1, summary2], references=[document, document]
    )
    print(harim_result)

    score_zs1 = model_zs.score([document], [summary1])
    score_conv1 = model_conv.score([document], [summary1])
    print(
        "[Summary 1] SummaCZS Score: %.3f; SummacConv score: %.3f"
        % (score_zs1["scores"][0], score_conv1["scores"][0])
    )  # [Summary 1] SummaCZS Score: 0.582; SummacConv score: 0.536

    score_zs2 = model_zs.score([document], [summary2])
    score_conv2 = model_conv.score([document], [summary2])
    print(
        "[Summary 2] SummaCZS Score: %.3f; SummacConv score: %.3f"
        % (score_zs2["scores"][0], score_conv2["scores"][0])
    )  # [Summary 2] SummaCZS Score: 0.877; SummacConv score: 0.709
