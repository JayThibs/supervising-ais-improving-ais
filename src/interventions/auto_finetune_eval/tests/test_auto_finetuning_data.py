# tests/test_auto_finetuning_data.py

import pytest
import random
import pandas as pd
import os
from unittest.mock import patch, MagicMock, mock_open
from transformers import PreTrainedModel
import sys
sys.path.append("../")
from auto_finetuning_data import (
    generate_ground_truths,
    generate_training_data,
    generate_dataset
)


@pytest.fixture
def mock_api_request():
    """
    A pytest fixture that can be used to mock out the make_api_request
    calls inside generate_ground_truths / generate_training_data / etc.
    Returns a mock function that you can configure to return custom responses.
    """
    with patch("auto_finetuning_data.make_api_request") as mock_req:
        # Default mock response
        mock_req.return_value = '["Mock response text"]'
        yield mock_req


@pytest.fixture
def mock_collect_dataset():
    """
    Fixture to mock out collect_dataset_from_api calls, which 
    are used to fetch JSON arrays of texts from the API.
    """
    with patch("auto_finetuning_data.collect_dataset_from_api") as mock_collect:
        mock_collect.return_value = ["Mock data 1", "Mock data 2"]
        yield mock_collect


@pytest.fixture
def mock_load_dataset():
    """
    Fixture to mock out HuggingFace load_dataset calls
    (used if use_truthful_qa = True).
    """
    with patch("auto_finetuning_data.load_dataset") as mock_ld:
        # Mock a minimal dataset structure
        fake_dataset = {
            "train": [
                {"question": "Fake Q1", "incorrect_answers": ["Wrong1"], "category": "Random"},
                {"question": "Fake Q2", "incorrect_answers": ["Wrong2"], "category": "Logical Falsehood"},
            ],
            "validation": [
                {"question": "Fake Q3", "incorrect_answers": ["Wrong3"], "category": "Random"}
            ]
        }
        mock_ld.return_value = fake_dataset
        yield mock_ld

@pytest.mark.usefixtures("mock_api_request", "mock_collect_dataset", "mock_load_dataset")
class TestAutoFinetuningData:

    def test_generate_ground_truths_no_truthful_qa(self):
        """
        Test that generate_ground_truths returns the correct number of items
        when not using TruthfulQA dataset.
        """
        ground_truths = generate_ground_truths(
            num_ground_truths=3,
            api_provider="openai",
            model_str="gpt-4-test",
            api_key="fake_api_key",
            use_truthful_qa=False
        )
        assert len(ground_truths) == 2, (
            "Mocked collect_dataset_from_api returned 2 items, "
            "so we expect exactly 2 ground truths."
        )
        assert all(isinstance(gt, str) for gt in ground_truths), "Ground truths should be strings"

    def test_generate_ground_truths_with_truthful_qa(self):
        """
        Test that using TruthfulQA dataset triggers the load_dataset mock,
        and that the function doesn't break with the minimal dataset.
        """
        ground_truths = generate_ground_truths(
            num_ground_truths=2,
            api_provider="anthropic",
            model_str="claude-test",
            api_key="fake_api_key",
            use_truthful_qa=True
        )
        # We configured collect_dataset_from_api to return ["Mock data 1", "Mock data 2"],
        # but for the truthful_qa path, it calls make_api_request on small prompts.
        # Let's just check we got the same shape as the random sample might produce:
        assert len(ground_truths) <= 2, "Should not exceed requested ground truths"
        assert all(isinstance(gt, str) for gt in ground_truths)

    def test_generate_training_data_basic(self):
        """
        Test generate_training_data with mocking of collect_dataset_from_api.
        """
        data = generate_training_data(
            ground_truth="AI thinks the moon is made of cheese",
            num_samples=3,
            api_provider="openai",
            model_str="gpt-4-test",
            api_key="fake_api_key"
        )
        assert len(data) == 2, (
            "Mock returns 2 items from collect_dataset_from_api, so we expect 2 training data samples."
        )
        assert all(isinstance(item, str) for item in data)

    @patch("auto_finetuning_data.batch_decode_texts", return_value=["decoded1", "decoded2"])
    def test_generate_dataset_minimal(self, mock_decode, mock_api_request, mock_collect_dataset):
        """
        Test generating a dataset from a small list of ground truths with minimal samples.
        Also checks if writing to CSV works when output_file_path is specified.
        """

        def mock_base_model():
            """
            Returns a MagicMock that mimics a HuggingFace PreTrainedModel.
            """
            mock_model = MagicMock(spec=PreTrainedModel)
            # Simulate config fields used in the CustomTrainer
            mock_model.config = MagicMock()
            mock_model.config.hidden_size = 128
            mock_model.config.num_attention_heads = 8
            mock_model.config.num_key_value_heads = 8

            # Simulate the code's attempt to print model.dtype and param.dtype
            mock_model.dtype = "float32"
            param_mock = MagicMock()
            param_mock.dtype = "float32"
            mock_model.parameters.return_value = iter([param_mock])

            # Optionally simulate presence/absence of quantization_config
            mock_model.quantization_config = None

            return mock_model

        def mock_tokenizer():
            """
            Returns a MagicMock that mimics a HuggingFace PreTrainedTokenizer,
            with proper dictionary return values for tokenization.
            """
            class TokenizerMock:
                """Custom mock class to ensure correct dictionary returns"""
                def __init__(self):
                    self.pad_token = "<pad>"
                    self.eos_token_id = 2
                    self.padding_side = "right"

                def __call__(self, texts, padding=None, truncation=None, max_length=None, **kwargs):
                    # Handle both single strings and lists of strings
                    if isinstance(texts, str):
                        texts = [texts]
                    
                    # Return properly formatted dictionary
                    return {
                        "input_ids": [[1, 2, 3] for _ in texts],
                        "attention_mask": [[1, 1, 1] for _ in texts]
                    }

            # Create our custom mock instead of using MagicMock
            return TokenizerMock()
        
        mock_base_model = mock_base_model()
        mock_tokenizer = mock_tokenizer()
        # We'll patch out batch_decode_texts so that we don't do any heavy decoding
        ground_truths = ["The AI likes cats", "The AI thinks it is a toaster"]
        df = generate_dataset(
            ground_truths=ground_truths,
            num_samples=2,
            api_provider="openai",
            model_str="gpt-4-test",
            api_key="fake_api_key",
            output_file_path="test_data.csv",
            num_base_samples_for_training=2,  # triggers the base_model decoding
            base_model=mock_base_model,
            tokenizer=mock_tokenizer
        )
        # Check the DataFrame contents
        assert isinstance(df, pd.DataFrame)
        assert not df.empty, "We expect a non-empty DataFrame with mock data"
        assert "ground_truth" in df.columns
        assert "train_text" in df.columns

        # Verify if the CSV got created
        assert os.path.exists("test_data.csv"), "Data CSV should have been written to disk"
        # Clean up
        os.remove("test_data.csv")

    def test_generate_dataset_no_output_file(self, mock_api_request, mock_collect_dataset):
        """
        Test that generate_dataset does not write a file if output_file_path is None.
        """
        ground_truths = ["AI is afraid of numbers."]
        df = generate_dataset(
            ground_truths=ground_truths,
            num_samples=1,
            api_provider="openai",
            model_str="gpt-4-test",
            api_key="fake_api_key",
            output_file_path=None,
            base_model=None,
            tokenizer=None
        )
        # Basic checks
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert df.iloc[0]["ground_truth"] == "AI is afraid of numbers."
        # Ensure no new file was created
        assert not os.path.exists("test_data.csv"), (
            "We never asked for an output file, so none should exist."
        )
