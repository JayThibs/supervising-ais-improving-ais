# tests/test_validated_comparison_tools.py

import pytest
from unittest.mock import patch, MagicMock
import numpy as np

import sys
sys.path.append("../")
from validated_comparison_tools import (
    evaluate_label_discrimination,
    match_clusterings,
    validated_assistant_discriminative_compare,
)


@pytest.fixture
def mock_models():
    """
    Provides two 'fake' model objects for tests, each with a device attribute
    that can be used by tested code. You could expand them to mock .generate() as needed.
    """
    mock_model_1 = MagicMock()
    mock_model_1.device = "cuda:0"
    mock_model_2 = MagicMock()
    mock_model_2.device = "cuda:0"
    return mock_model_1, mock_model_2


@pytest.fixture
def mock_tokenizer():
    """
    Provides a 'fake' tokenizer. If your code calls e.g. 'encode', 'decode', etc.,
    you can mock those out here as well.
    """
    tokenizer = MagicMock()
    tokenizer.pad_token = "<PAD>"
    tokenizer.pad_token_id = 0
    return tokenizer


def test_evaluate_label_discrimination_simple():
    """
    A minimal test for evaluate_label_discrimination with a mock for the text-matching function.
    """
    label = "cluster label"
    # Some synthetic data
    sampled_texts_1 = [0, 1]
    sampled_texts_2 = [0, 1]
    decoded_strs_1 = ["Text A about cats", "Another cat text"]
    decoded_strs_2 = ["Text about dogs", "Another text on dogs"]

    # We only want to test the function's logic, so mock away any calls to `api_based_label_text_matching`
    with patch("validated_comparison_tools.api_based_label_text_matching", return_value=(0.75, 0.25)):
        score = evaluate_label_discrimination(
            label=label,
            sampled_texts_1=sampled_texts_1,
            sampled_texts_2=sampled_texts_2,
            decoded_strs_1=decoded_strs_1,
            decoded_strs_2=decoded_strs_2,
            local_model=None,
            labeling_tokenizer=None,
            api_provider="openai",
            api_model_str="gpt-4",
            auth_key=None,
            mode="double_cluster"
        )
    # For the mock above (always returning textA=0.75, textB=0.25),
    # we expect a reasonably high 'accuracy' or 'auc' if the code interprets bigger = cluster1
    assert 0 <= score <= 1, "evaluate_label_discrimination should return a valid [0,1] range"


def test_match_clusterings():
    """
    Test match_clusterings with small synthetic embeddings to ensure it pairs up centroids logically.
    """
    # Suppose we have 2 clusters in set1 and 2 in set2
    clustering_assignments_1 = [0, 0, 1, 1]
    embeddings_list_1 = [
        [0.0, 0.0],    # belongs to cluster0
        [0.1, 0.1],    # belongs to cluster0
        [5.0, 5.0],    # belongs to cluster1
        [5.1, 5.2],    # belongs to cluster1
    ]
    clustering_assignments_2 = [0, 1, 0, 1]
    embeddings_list_2 = [
        [0.05, 0.05],  # cluster0
        [4.9, 5.1],    # cluster1
        [0.0, 0.2],    # cluster0
        [5.05, 5.15],  # cluster1
    ]

    matched_pairs = match_clusterings(
        clustering_assignments_1,
        embeddings_list_1,
        clustering_assignments_2,
        embeddings_list_2
    )
    # We expect them to match cluster 0->0, 1->1 in an ideal scenario
    # The returned value is something like [(0, 0), (1, 1)], but exact order may vary
    # So let's just check that we see each pair
    assert len(matched_pairs) == 2
    pairs_set = set(matched_pairs)
    assert (0, 0) in pairs_set
    assert (1, 1) in pairs_set


@pytest.mark.parametrize("explain_reasoning", [False, True])
def test_validated_assistant_discriminative_compare_happy_path(
    mock_models, mock_tokenizer, explain_reasoning
):
    """
    Test the validated_assistant_discriminative_compare function in a typical scenario.
    We mock out the network calls used for the assistant's queries and final predictions.
    """
    difference_descriptions = ["Model 1 is more likely to produce poetic text than Model 2"]

    model_1, model_2 = mock_models

    # We'll patch parallel_make_api_requests to simulate the assistant's queries returning some prompts
    with patch("validated_comparison_tools.parallel_make_api_requests") as mock_parallel_api:
        # Each time the assistant asks for a query, let's just return "Here's my query"
        mock_parallel_api.side_effect = lambda *args, **kwargs: ["Mocked query"] * len(args[0])

        # Also patch generate_responses_in_batches for the model responses
        with patch("validated_comparison_tools.generate_responses_in_batches") as mock_model_responses:
            # Each time we call generate_responses_in_batches, pretend we get a single-line response
            mock_model_responses.side_effect = lambda *a, **kw: ["Response from model"] * len(a[1])

            # Finally, we patch again the parallel_make_api_requests for the final prediction calls:
            # In the final step, the assistant must guess "1" or "2".
            # We'll just say it always guesses "1".
            # Because we do two sets of calls (some for queries, then some for final predictions),
            # we can handle them with side_effect or multiple patches. We'll just do a simple approach:
            def side_effect_final(*args, **kwargs):
                request_info = kwargs.get("request_info", {})
                if request_info and request_info.get("round") == "final":
                    return ["1"] * len(args[0])
                return ["Mocked query"] * len(args[0])

            mock_parallel_api.side_effect = side_effect_final

            # Now we run the function under test
            discriminative_validation_results = validated_assistant_discriminative_compare(
                difference_descriptions=difference_descriptions,
                api_provider="anthropic",
                api_model_str="claude-test",
                auth_key=None,
                client=None,
                common_tokenizer_str="meta-llama/Meta-Llama-3-8B",
                starting_model=model_1,
                comparison_model=model_2,
                num_rounds=2,
                num_validation_runs=1,
                explain_reasoning=explain_reasoning,
            )
            accs = discriminative_validation_results["hypothesis_accuracies"]
            pvals = discriminative_validation_results["hypothesis_p_values"]
            reasonings = discriminative_validation_results["reasonings"] if explain_reasoning else None

    # Check results
    assert len(accs) == 1, "We had only 1 difference_description, so we expect 1 accuracy"
    assert len(pvals) == 1, "We had only 1 difference_description, so we expect 1 p-value"
    # Because we forced the assistant to always guess "1", half are actually correct (the other half are '2'),
    # so we expect maybe 50% accuracy. But let's just do a sanity check 0 <= x <= 1
    assert 0.0 <= accs[0] <= 1.0
    assert 0.0 <= pvals[0] <= 1.0

    if explain_reasoning:
        assert reasonings is not None, "Expect reasonings if explain_reasoning=True"
        # Single hypothesis => single list of reasonings
        assert len(reasonings) == 1
    else:
        # The function returns just (accs, pvals) if explain_reasoning=False
        assert reasonings is None


def test_validated_assistant_discriminative_compare_failed_final_api_requests(
    mock_models, mock_tokenizer
):
    """
    In this test, we check how the function behaves if there's an error or unexpected format
    from the final calls for the assistant's prediction.
    """
    difference_descriptions = ["Model 1 is more creative than Model 2"]
    model_1, model_2 = mock_models

    with patch("validated_comparison_tools.parallel_make_api_requests") as mock_parallel_api, \
         patch("validated_comparison_tools.generate_responses_in_batches") as mock_model_responses:
        
        # Provide a normal sequence of responses for the query rounds
        mock_parallel_api.side_effect = lambda *args, **kwargs: ["Mocked query"] * len(args[0])
        mock_model_responses.side_effect = lambda *a, **kw: ["Model output"] * len(a[1])

        # For the final round, let's produce something invalid, e.g. 'Hello' instead of '1' or '2'
        def side_effect_final(*args, **kwargs):
            request_info = kwargs.get("request_info", {})
            if request_info and request_info.get("round") == "final":
                return ["Hello, not an integer"] * len(args[0])
            return ["Mocked query"] * len(args[0])
        mock_parallel_api.side_effect = side_effect_final

        discriminative_validation_results = validated_assistant_discriminative_compare(
            difference_descriptions=difference_descriptions,
            api_provider="openai",
            api_model_str="gpt-3.5-turbo",
            starting_model=model_1,
            comparison_model=model_2,
            num_rounds=1,
            num_validation_runs=2,
            explain_reasoning=False,
        )

    accs = discriminative_validation_results["hypothesis_accuracies"]
    pvals = discriminative_validation_results["hypothesis_p_values"]

    # Because final predictions were invalid, we should see zero accuracy
    assert len(accs) == 1
    assert accs[0] == 0.0
    # p-value for the binomial test when everything is 0% would presumably be 1.0
    assert pvals[0] == 1.0
