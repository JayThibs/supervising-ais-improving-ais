# tests/test_auto_finetuning_interp.py

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

import sys
sys.path.append("../")

from auto_finetuning_interp import (
    setup_interpretability_method,
    apply_interpretability_method,
    get_individual_labels,
    generate_table_output,
    apply_interpretability_method_1_to_K
)

###############################################################################
# Example fixtures for lightweight model and tokenizer mocks
###############################################################################


@pytest.fixture
def mock_base_model():
    """
    Returns a mock base model object configured to handle batch generation properly.
    """
    model = MagicMock(name="MockBaseModel")
    
    # Create mock tensor outputs for generate()
    mock_outputs = MagicMock()
    model.generate.return_value = mock_outputs
    
    # Set device attribute to match what's expected
    model.device = "cpu"
    
    # Set a valid model identifier as the name_or_path
    model.name_or_path = "meta-llama/Meta-Llama-3-8B"
    
    return model


@pytest.fixture
def mock_finetuned_model():
    """
    Returns a mock finetuned model object configured to handle batch generation properly.
    """
    model = MagicMock(name="MockFinetunedModel")
    # Create mock tensor outputs for generate()
    mock_outputs = MagicMock()
    model.generate.return_value = mock_outputs

    model.device = "cpu"

    # Set a valid model identifier as the name_or_path
    model.name_or_path = "meta-llama/Meta-Llama-3-8B"

    return model


@pytest.fixture
def mock_tokenizer():
    """
    Returns a mock tokenizer configured to handle the complete encode-decode cycle.
    """
    tokenizer = MagicMock(name="MockTokenizer")
    
    # Configure tokenizer attributes
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 2
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token = "</s>"
    
    # Simulate calling the tokenizer
    def tokenizer_call(*args, **kwargs):
        import torch
        mock_encoding = MagicMock()
        mock_encoding.input_ids = torch.tensor([[1, 2, 3, 4]])
        mock_encoding.to = lambda device: mock_encoding
        return mock_encoding
    tokenizer.__call__ = tokenizer_call
    
    # Configure batch_decode to return some short sample texts
    def batch_decode(*args, **kwargs):
        return [f"decoded_text_{i}" for i in range(6)]
    tokenizer.batch_decode = batch_decode
    
    return tokenizer


###############################################################################
# Mock for read_past_embeddings_or_generate_new to skip real embedding calls
###############################################################################

@pytest.fixture
def mock_embeddings_patch():
    with patch("auto_finetuning_interp.read_past_embeddings_or_generate_new") as mock_embed:
        # Return small dummy embeddings (2D vectors).
        mock_embed.return_value = np.array([
            [0.1, 0.2],
            [0.2, 0.1],
            [0.8, 0.9],
            [1.0, 1.1],
            [0.4, 0.6],
            [0.2, 0.25],
        ])
        yield mock_embed


###############################################################################
# Fixture to patch ALL references to make_api_request/parallel_make_api_requests
# in validated_comparison_tools and auto_finetuning_helpers
###############################################################################

@pytest.fixture
def mock_all_api_requests():
    """
    Patches all references to make_api_request and parallel_make_api_requests
    in validated_comparison_tools and auto_finetuning_helpers, so we never
    accidentally invoke real external calls.
    """
    with patch("validated_comparison_tools.make_api_request") as v_m:
        with patch("validated_comparison_tools.parallel_make_api_requests") as v_pm:
            with patch("auto_finetuning_helpers.make_api_request") as h_m:
                with patch("auto_finetuning_helpers.parallel_make_api_requests") as h_pm:
                    # Example return values that your code might expect:
                    v_m.return_value = '{"similarity_score": 88}'
                    v_pm.return_value = ['{"similarity_score": 88}']
                    h_m.return_value = '{"similarity_score": 88}'
                    h_pm.return_value = ['{"similarity_score": 88}']
                    yield {
                        "val_make_api": v_m,
                        "val_par_make_api": v_pm,
                        "help_make_api": h_m,
                        "help_par_make_api": h_pm
                    }

###############################################################################
# Tests start here
###############################################################################


def test_setup_interpretability_method_minimal(
    mock_base_model, mock_finetuned_model, mock_tokenizer, 
    mock_embeddings_patch, mock_all_api_requests
):
    """
    Test setup_interpretability_method with proper batch decoding simulation.
    Verifies that no real external calls are made if the code doesn't need them.
    """
    with patch('auto_finetuning_interp.batch_decode_texts') as mock_batch_decode:
        mock_batch_decode.side_effect = lambda model, tokenizer, prefixes, n, **kwargs: [
            f"text_{i}" for i in range(6)
        ]
        
        result = setup_interpretability_method(
            base_model=mock_base_model,
            finetuned_model=mock_finetuned_model,
            tokenizer=mock_tokenizer,
            n_decoded_texts=6,
            local_embedding_model_str="nvidia/NV-Embed-v1",
            auth_key="fake_key",
            client=MagicMock(),
            cluster_method="kmeans",
            n_clusters=2,
            run_prefix="test_run_prefix"
        )

    assert "base_clustering" in result
    assert "finetuned_clustering" in result
    assert "base_embeddings" in result
    assert "finetuned_embeddings" in result
    assert "base_decoded_texts" in result
    assert "finetuned_decoded_texts" in result

    # Because we forced 2 clusters, verify we see 2 or fewer unique cluster IDs
    base_unique = set(result["base_clustering_assignments"])
    finetuned_unique = set(result["finetuned_clustering_assignments"])
    assert len(base_unique) <= 2
    assert len(finetuned_unique) <= 2

    assert len(result["base_decoded_texts"]) == 6
    assert len(result["finetuned_decoded_texts"]) == 6

    # Embeddings function was called for both base & finetuned
    assert mock_embeddings_patch.call_count == 2

    # Check that no real external calls were made if the code doesn't invoke them
    assert mock_all_api_requests["val_make_api"].call_count == 0
    assert mock_all_api_requests["help_make_api"].call_count == 0


def test_apply_interpretability_method_small(
    mock_base_model, mock_finetuned_model, mock_tokenizer,
    mock_embeddings_patch, mock_all_api_requests
):
    """
    Test apply_interpretability_method, ensuring we patch external API calls 
    and handle minimal data.
    """
    with patch("auto_finetuning_interp.match_clusterings", return_value=[(0, 0)]):
        with patch("auto_finetuning_interp.get_validated_contrastive_cluster_labels") as mock_contrastive:
            mock_contrastive.return_value = {
                "cluster_pair_scores": {(0, 0): {"label0": 0.85}},
                "p_values": {(0, 0): {"label0": 0.05}}
            }

            final_hypotheses = apply_interpretability_method(
                base_model=mock_base_model,
                finetuned_model=mock_finetuned_model,
                tokenizer=mock_tokenizer,
                n_decoded_texts=6,
                local_embedding_model_str="fake-embed-model",
                api_provider="anthropic",
                api_model_str="fake-model",
                auth_key="fake-key",
                client=MagicMock(),
                cluster_method="kmeans",
                n_clusters=2
            )

            assert isinstance(final_hypotheses, list)
            assert len(final_hypotheses) > 0

    # If the code calls make_api_request internally, we can check the count here
    # e.g., assert mock_all_api_requests["val_make_api"].call_count >= 1


def test_get_individual_labels_basic(
    mock_base_model, mock_tokenizer, mock_all_api_requests
):
    """
    Testing get_individual_labels with external API calls mocked out.
    """
    from unittest.mock import patch

    decoded_texts = ["Text A", "Text B", "Text C", "Text D"]
    clustering_assignments = [0, 0, 1, 1]

    with patch("auto_finetuning_interp.get_cluster_labels_random_subsets") as mock_rand:
        mock_rand.return_value = (
            {
                0: ["Label1_cluster0", "Label2_cluster0"],
                1: ["Label1_cluster1", "Label2_cluster1"]
            },
            {}
        )

        labels_dict = get_individual_labels(
            decoded_strs=decoded_texts,
            clustering_assignments=clustering_assignments,
            local_model=None,
            labeling_tokenizer=mock_tokenizer,
            api_provider="anthropic",  
            api_model_str="fake-model",
            auth_key="fake-key",
            client=MagicMock(),
            device="cpu",
            sampled_texts_per_cluster=2,
            generated_labels_per_cluster=2
        )

        assert len(labels_dict) == 2
        for cluster_id, (best_label, best_score) in labels_dict.items():
            assert isinstance(best_label, str)
            assert isinstance(best_score, float)

    # For actual calls: 
    # assert mock_all_api_requests["val_make_api"].call_count >= 1


def test_apply_interpretability_method_1_to_K(
    mock_base_model, mock_finetuned_model, mock_tokenizer,
    mock_embeddings_patch, mock_all_api_requests
):
    """
    Test apply_interpretability_method_1_to_K with minimal data,
    ensuring external APIs are mocked.
    """
    # First, improve our API response mocking
    def mock_api_response(*args, **kwargs):
        # For generating cluster labels and similar tasks
        if not kwargs.get('request_info'):
            return ['{"similarity_score": 88}']
            
        # Get the pipeline stage and batch info
        pipeline_stage = kwargs['request_info'].get('pipeline_stage')
        batch = kwargs['request_info'].get('batch')
        
        # Handle discriminative validation requests
        if pipeline_stage == "discriminative_validation":
            if batch in ['initial_queries', 'follow_up_queries']:
                queries = ["What would you say about love?"] * len(args[0])  # Match number of prompts
                return queries
            elif batch == 'final_predictions':
                return ["1"] * len(args[0])  # Match number of test cases
        
        # Default response for other cases
        return ['{"similarity_score": 88}']

    # Set up our mocks
    mock_all_api_requests["val_par_make_api"].side_effect = mock_api_response
    mock_all_api_requests["help_par_make_api"].side_effect = mock_api_response
    
    with patch("auto_finetuning_interp.build_contrastive_K_neighbor_similarity_graph") as mock_graph_builder:
        mock_graph = MagicMock()
        mock_graph.neighbors.side_effect = lambda x: []  # No neighbors
        mock_graph_builder.return_value = mock_graph

        with patch("pickle.dump") as mock_pickle:
            # Provide labels for both clusters (0 and 1)
            mock_labels = {
                0: ("Label0", 0.75),
                1: ("Label1", 0.80)
            }
            
            # Also patch out get_individual_labels
            with patch("auto_finetuning_interp.get_individual_labels", return_value=mock_labels):
                # Initialize test parameters
                results, table_output, validated_hypotheses = apply_interpretability_method_1_to_K(
                    base_model=mock_base_model,
                    finetuned_model=mock_finetuned_model,
                    tokenizer=mock_tokenizer,
                    n_decoded_texts=6,
                    api_provider="anthropic",
                    api_model_str="fake_model_str",
                    auth_key="fake-key",
                    client=MagicMock(),
                    local_embedding_model_str="fake-embed-model",
                    cluster_method="kmeans",
                    n_clusters=2,
                    K=1
                )

                # Add assertions
                assert isinstance(results, dict)
                assert "base_clusters" in results
                assert len(results["base_clusters"]) == 2
                assert isinstance(table_output, str)
                assert isinstance(validated_hypotheses, list)

    # Check pickle.dump was called
    mock_pickle.assert_called_once()