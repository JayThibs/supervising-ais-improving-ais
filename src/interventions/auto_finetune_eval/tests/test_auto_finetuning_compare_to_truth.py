# File: tests/test_auto_finetuning_compare_to_truth.py

import json
import pytest
from unittest.mock import patch, MagicMock

# Import the functions under test
import sys
sys.path.append("../")
from auto_finetuning_compare_to_truth import compare_hypotheses, compare_and_score_hypotheses


@pytest.fixture
def mock_api_response_ok():
    """
    A helper fixture that returns a valid JSON response with a key 'similarity_score'.
    """
    return json.dumps({"similarity_score": 88})


@pytest.fixture
def mock_api_response_bad():
    """
    A helper fixture that returns a malformed or otherwise unexpected JSON response.
    """
    return "Oops! This isn't valid JSON. { nonsense"


@pytest.fixture
def mock_api_response_cluster2():
    """
    A helper fixture simulating a scenario where the ground truth triggers the 'Cluster 2' condition,
    and returns a valid JSON with a score.
    """
    return json.dumps({"similarity_score": 100})


@pytest.mark.parametrize("ground_truth, discovered_hypothesis, contains_cluster2", [
    ("Some ground truth about cats", "A discovered hypothesis about cats", False),
    ("The AI believes Cluster 2 is malicious", "Discovered cluster 2 text", True),
])
def test_compare_hypotheses_normal_flow(
    ground_truth, 
    discovered_hypothesis, 
    contains_cluster2,
    mock_api_response_ok
):
    """
    Test normal flows through compare_hypotheses, verifying that we handle JSON properly
    and return the correct float.
    """

    # We'll patch make_api_request to return mock_api_response_ok
    with patch("auto_finetuning_compare_to_truth.make_api_request") as mock_request:
        mock_request.return_value = mock_api_response_ok

        # Call the function under test
        result = compare_hypotheses(
            ground_truth=ground_truth,
            discovered_hypothesis=discovered_hypothesis,
            api_provider="openai",
            model_str="gpt-3.5",
        )

        # If the function ran successfully, we expect the result to be 88.0
        assert result == 88.0

        # Check prompt logic: if "Cluster 2" is in ground_truth, a different prompt path is used
        call_args_list = mock_request.call_args_list
        assert len(call_args_list) == 1
        args, kwargs = call_args_list[0]
        # The prompt is in args[0]. We can do minimal checks:
        if contains_cluster2:
            assert "Compare the ground truth statement with the description of cluster 2" in args[0]
        else:
            assert "Compare the following ground truth statement with a discovered hypothesis" in args[0]


def test_compare_hypotheses_malformed_json(mock_api_response_bad):
    """
    Test that if the API returns malformed JSON, compare_hypotheses doesn't throw an error
    but defaults the score to 0.
    """
    with patch("auto_finetuning_compare_to_truth.make_api_request") as mock_request:
        mock_request.return_value = mock_api_response_bad

        score = compare_hypotheses(
            ground_truth="We test a weird response",
            discovered_hypothesis="Testing a weird response",
            api_provider="openai",
            model_str="fake-model"
        )
        assert score == 0.0, "Expected score=0.0 when JSON decode fails"


def test_compare_hypotheses_no_similarity_key():
    """
    If the JSON is valid but missing the 'similarity_score' field, ensure the fallback is 0.
    """
    with patch("auto_finetuning_compare_to_truth.make_api_request") as mock_request:
        mock_request.return_value = '{"some_other_key": 75}'
        
        score = compare_hypotheses(
            ground_truth="Some GT",
            discovered_hypothesis="Some Hyp",
            api_provider="openai",
            model_str="fake-model"
        )
        assert score == 0.0, "Expected fallback to 0 if 'similarity_score' is missing"


# -------------------------------------------------------------
# Tests for compare_and_score_hypotheses
# -------------------------------------------------------------
@pytest.mark.parametrize("match_by_embedding, match_by_bleu", [
    (False, False),
    (True, False),
    (False, True),
])
def test_compare_and_score_hypotheses_basic(
    mock_api_response_ok,
    match_by_embedding,
    match_by_bleu
):
    """
    Test compare_and_score_hypotheses under different matching modes:
      - Full pairwise comparison
      - match_by_embedding
      - match_by_bleu

    We'll mock out the calls to make_api_request so we can verify the final results
    quickly, ignoring embedding/bleu logic details.
    """
    with patch("auto_finetuning_compare_to_truth.make_api_request") as mock_request:
        # We'll always return the same JSON so that each pair comparison yields 88.0
        mock_request.return_value = mock_api_response_ok

        # We'll pass in 2 ground truths and 2 discovered_hypotheses
        ground_truths = ["The AI believes in pillows", "The AI likes cats"]
        discovered_hyps = ["Hyp about pillows", "Hyp about cats"]

        results = compare_and_score_hypotheses(
            ground_truths,
            discovered_hyps,
            api_provider="anthropic",
            model_str="claude-2",
            match_by_embedding=match_by_embedding,
            match_by_bleu=match_by_bleu
        )

        # results is a dict containing: individual_scores, average_score, etc.
        assert "individual_scores" in results
        assert "average_score" in results
        assert "max_score" in results
        assert "min_score" in results
        assert "matched_hypotheses" in results

        # Because we return 88.0 for all comparisons, let's see how many comparisons we do:
        if match_by_embedding or match_by_bleu:
            # If matching by embedding or BLEU, it does a bipartite match of size len(ground_truths)
            # So there should be 2 comparisons total (one per GT).
            expected_len = len(ground_truths)
        else:
            # Pairwise: we do len(GT) * len(Hyps) = 2*2 = 4 comparisons
            expected_len = len(ground_truths) * len(discovered_hyps)

        assert len(results["individual_scores"]) == expected_len
        # All of them should be 88.0
        for s in results["individual_scores"]:
            assert s == 88.0

        # Then the average, min, max, matched
        assert results["average_score"] == 88.0
        assert results["max_score"] == 88.0
        assert results["min_score"] == 88.0

        # matched_hypotheses is the count of scores > 80
        assert results["matched_hypotheses"] == expected_len


def test_compare_and_score_hypotheses_with_embedding_mocks():
    """
    Specifically test the code branch that calls read_past_embeddings_or_generate_new
    when match_by_embedding=True. We'll mock that out to avoid real embeddings logic.
    """
    ground_truths = ["GT1", "GT2"]
    hyps = ["HYP1", "HYP2"]

    # Patch read_past_embeddings_or_generate_new to return dummy embeddings
    with patch("auto_finetuning_compare_to_truth.read_past_embeddings_or_generate_new") as mock_embed:
        mock_embed.return_value = [
            [0.1, 0.2, 0.3],  # embedding for ground_truth[0]
            [0.4, 0.5, 0.6],  # embedding for ground_truth[1]
            [0.7, 0.8, 0.9],  # embedding for discovered_hyps[0]
            [0.2, 0.2, 0.2],  # embedding for discovered_hyps[1]
        ]

        # Also patch make_api_request
        with patch("auto_finetuning_compare_to_truth.make_api_request") as mock_request:
            mock_request.return_value = json.dumps({"similarity_score": 90})

            results = compare_and_score_hypotheses(
                ground_truths,
                hyps,
                api_provider="openai",
                model_str="gpt-3.5",
                match_by_embedding=True,
                match_by_embedding_model="Fake-Embedding-Model"
            )

            # Because we do linear_sum_assignment on the pairwise embeddings, we won't check
            # correctness of that logic deeply, but let's see if it yields 2 comparisons in total
            assert len(results["individual_scores"]) == 2
            # They should each be 90 from the patched response
            for val in results["individual_scores"]:
                assert val == 90.0

            # check that read_past_embeddings_or_generate_new was called once
            mock_embed.assert_called_once()
            # check that make_api_request was called exactly 2 times
            assert mock_request.call_count == 2


def test_compare_and_score_hypotheses_no_hypotheses():
    """
    Edge case: if discovered_hypotheses is empty, we might return empty lists or get a ZeroDivisionError.
    Ensure we handle gracefully.
    """
    with patch("auto_finetuning_compare_to_truth.make_api_request") as mock_request:
        mock_request.return_value = '{"similarity_score": 99}'

        # If we pass an empty discovered_hypotheses, we'd expect no comparisons
        ground_truths = ["GT1", "GT2"]
        discovered_hyps = []

        results = compare_and_score_hypotheses(
            ground_truths,
            discovered_hyps,
            api_provider="anthropic",
            model_str="claude-instant"
        )

        # We expect that "individual_scores" is empty, and we have to see how the code handles that
        assert len(results["individual_scores"]) == 0
        assert results["average_score"] == 0  # or we might define some fallback
        assert results["max_score"] == 0
        assert results["min_score"] == 0
        assert results["matched_hypotheses"] == 0
