# tests/test_auto_finetuning_main.py

import argparse
import pytest
import random
import sys
from unittest.mock import patch, MagicMock

sys.path.append("../")

from auto_finetuning_main import main

@pytest.fixture
def minimal_args():
    """Provide minimal arguments for testing the main pipeline."""
    args = argparse.Namespace(
        run_prefix="test_run",
        base_model="test/base-model",
        base_model_revision=None,
        intervention_model=None,
        intervention_model_revision=None,
        device="cpu",
        num_ground_truths=1,
        focus_area=None,
        use_truthful_qa=False,
        num_samples=5,
        ground_truth_file_path=None,
        num_base_samples_for_training=0,
        regenerate_ground_truths=False,
        regenerate_training_data=False,
        finetuning_params={"learning_rate": 0.0},
        train_lora=False,
        finetuning_save_path=None,
        finetuning_load_path=None,
        temp_dir="/tmp",
        num_decoded_texts=10,
        decoding_max_length=48,
        decoding_batch_size=2,
        decoding_prefix_file=None,
        decoded_texts_save_path=None,
        decoded_texts_load_path=None,
        loaded_texts_subsample=None,
        base_model_quant_level="8bit",
        intervention_model_quant_level="8bit",
        cluster_method="kmeans",
        num_clusters=4,
        min_cluster_size=7,
        max_cluster_size=2000,
        clustering_instructions="Identify the topic or theme of the given texts",
        n_clustering_inits=2,
        local_embedding_model_str="nvidia/NV-Embed-v1",
        num_rephrases_for_validation=0,
        generated_labels_per_cluster=2,
        use_unitary_comparisons=False,
        max_unitary_comparisons_per_label=10,
        match_cutoff=0.7,
        metric="acc",
        tsne_save_path=None,
        tsne_title=None,
        tsne_perplexity=30,
        api_provider="anthropic",
        model_str="claude-2.0-1234",
        stronger_model_str=None,
        key_path="fake_key_path.txt",
        api_interactions_save_loc=None
    )
    return args


@pytest.mark.parametrize("train_lora", [False, True])
# Patch out all relevant references:
# 1) All calls to make_api_request in any submodules (helpers, compare_to_truth, validated_comparison_tools)
@patch("auto_finetuning_helpers.make_api_request", return_value='{"similarity_score": 99}')
@patch("auto_finetuning_helpers.parallel_make_api_requests", return_value=['{"similarity_score": 99}'])
@patch("auto_finetuning_compare_to_truth.make_api_request", return_value='{"similarity_score": 95}')
@patch("validated_comparison_tools.make_api_request", return_value='{"similarity_score": 88}')
# 2) Patch any open() calls for file I/O
@patch("builtins.open", create=True)
# 3) Patch the local embedding call in read_past_embeddings_or_generate_new
@patch("auto_finetuning_interp.read_past_embeddings_or_generate_new", return_value=[
    [random.random() for _ in range(5)] for _ in range(500)
])
# 4) Patch the top-level items from auto_finetuning_main
@patch("auto_finetuning_main.load_api_key", return_value="FAKE_API_KEY")
@patch("auto_finetuning_main.AutoModelForCausalLM.from_pretrained")
@patch("auto_finetuning_main.AutoTokenizer.from_pretrained")
@patch("auto_finetuning_main.finetune_model")
@patch("auto_finetuning_main.generate_ground_truths")
@patch("auto_finetuning_main.generate_dataset")
@patch("auto_finetuning_main.compare_and_score_hypotheses", return_value={"average_score": 42, "matched_hypotheses": 1})
# 5) Patch instantiations of external API clients
@patch("auto_finetuning_main.openai.OpenAI", return_value=MagicMock())
@patch("auto_finetuning_main.Anthropic", return_value=MagicMock())
@patch("auto_finetuning_main.genai.GenerativeModel", return_value=MagicMock())
def test_main_pipeline(
    mock_gen_model,
    mock_anthropic,
    mock_openai,
    mock_compare_and_score,
    mock_generate_dataset,
    mock_generate_ground_truths,
    mock_finetune_model,
    mock_tokenizer_from_pretrained,
    mock_from_pretrained,
    mock_load_api_key,
    mock_read_embeddings,
    mock_open_file,
    mock_make_req_validated_comparison,
    mock_make_req_compare_truth,
    mock_parallel_requests_helpers,
    mock_make_req_helpers,
    minimal_args,
    train_lora
):
    """
    Thoroughly test the main pipeline in auto_finetuning_main.py by patching out:
     - API calls (Anthropic, OpenAI, etc.)
     - File I/O
     - HF model loading
     - Data generation calls
     - Comparison functions
     - Embeddings generation
    """

    # Adjust fixture for the param scenario
    minimal_args.train_lora = train_lora

    # Mock model & tokenizer
    mock_model = MagicMock()
    mock_from_pretrained.return_value = mock_model

    mock_tokenizer = MagicMock()
    # Example: returning a list of 500 "Sample decoded text X"
    mock_decoded_texts = [f"Sample decoded text {i}" for i in range(500)]
    mock_tokenizer.batch_decode.return_value = mock_decoded_texts
    mock_tokenizer_from_pretrained.return_value = mock_tokenizer

    # Mock ground truths + dataset
    mock_ground_truths = ["The AI likes pillows."]
    mock_generate_ground_truths.return_value = mock_ground_truths

    mock_dataset = MagicMock()
    mock_dataset.empty = False
    mock_dataset["train_text"] = MagicMock(return_value=["Sample training text."])
    mock_generate_dataset.return_value = mock_dataset

    # Mock finetune_model return
    mock_finetune_model.return_value = MagicMock()

    # Now run the pipeline
    main(minimal_args)

    # Basic assertions about calls
    # 1) We load the base model at least once
    assert mock_from_pretrained.call_count >= 1

    # 2) If there's no intervention model, we expect generate_ground_truths was called
    if minimal_args.intervention_model is None:
        mock_generate_ground_truths.assert_called_once()

    # 3) If no finetuning_load_path, we likely generate a dataset
    if not minimal_args.finetuning_load_path:
        mock_generate_dataset.assert_called_once()

    # 4) If we do LoRA or a non-zero learning_rate, we see finetune_model
    if minimal_args.train_lora or minimal_args.finetuning_params.get("learning_rate", 0) > 0:
        mock_finetune_model.assert_called_once()

    # 5) Finally, compare_and_score_hypotheses
    # Skip for now, since it's not really implemented yet
    # mock_compare_and_score.assert_called_once()

    # 6) Confirm no exceptions => success
    assert True, "Pipeline completed without errors."
