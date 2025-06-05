# tests/test_auto_finetuning_train.py

import pytest
from unittest.mock import patch, MagicMock, call
from transformers import PreTrainedModel, PreTrainedTokenizer
import sys
sys.path.append("../")
from auto_finetuning_train import finetune_model


@pytest.fixture
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

@pytest.fixture
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

@pytest.fixture
def sample_training_data():
    """Returns minimal training data for testing."""
    return ["Hello world", "Testing the finetuning function", "Another sample text"]

@pytest.fixture
def default_finetuning_params():
    """Returns typical minimal config for finetuning_params."""
    return {
        "num_epochs": 1,
        "batch_size": 2,
        "device_batch_size": 1,
        "max_length": 16,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "learning_rate": 5e-5,
        "save_steps": 50,
    }

@patch("auto_finetuning_train.Trainer")
@patch("auto_finetuning_train.prepare_model_for_kbit_training")
@patch("auto_finetuning_train.get_peft_model")
def test_finetune_model_lora(
    mock_get_peft_model,
    mock_prepare_model_for_kbit_training,
    mock_trainer_class,
    mock_base_model,
    mock_tokenizer,
    sample_training_data,
    default_finetuning_params
):
    """
    Test that finetune_model correctly sets up LoRA, calls Trainer,
    and returns the merged model when train_lora=True.
    """
    # Mock the trainer instance
    mock_trainer = MagicMock()
    mock_trainer.train_dataset = MagicMock()
    mock_trainer.train.return_value = None  # Simulate a normal run
    # The "model" attribute after training
    mock_trainer.model = MagicMock()
    # Merging step
    mock_trainer.model.merge_and_unload.return_value = MagicMock()

    mock_trainer_class.return_value = mock_trainer

    # Actually call the function
    finetuned_model = finetune_model(
        base_model=mock_base_model,
        tokenizer=mock_tokenizer,
        training_data=sample_training_data,
        finetuning_params=default_finetuning_params,
        train_lora=True
    )

    # Assertions:
    # 1) Should have prepared model for k-bit training
    mock_prepare_model_for_kbit_training.assert_called_once_with(mock_base_model)

    # 2) Should have created the PEFT model
    mock_get_peft_model.assert_called_once()

    # 3) Trainer should have been constructed and train() called
    assert mock_trainer_class.call_count == 1
    mock_trainer.train.assert_called_once()

    # 4) Because train_lora=True, it merges LoRA weights
    mock_trainer.model.merge_and_unload.assert_called_once()
    assert finetuned_model == mock_trainer.model.merge_and_unload.return_value

@patch("auto_finetuning_train.Trainer")
def test_finetune_model_no_lora(
    mock_trainer_class,
    mock_base_model,
    mock_tokenizer,
    sample_training_data,
    default_finetuning_params
):
    """
    Test that when train_lora=False, the function does not do LoRA manipulations,
    and returns the final trainer.model directly.
    """
    mock_trainer = MagicMock()
    mock_trainer.train_dataset = MagicMock()
    mock_trainer.train.return_value = None
    mock_trainer.model = MagicMock()
    mock_trainer_class.return_value = mock_trainer

    # Set train_lora=False
    finetuned_model = finetune_model(
        base_model=mock_base_model,
        tokenizer=mock_tokenizer,
        training_data=sample_training_data,
        finetuning_params=default_finetuning_params,
        train_lora=False
    )

    # Should not call LoRA methods
    mock_trainer.train.assert_called_once()
    mock_trainer.model.merge_and_unload.assert_not_called()

    # The returned model is just trainer.model
    assert finetuned_model == mock_trainer.model

@patch("auto_finetuning_train.Trainer")
def test_finetune_model_empty_data(
    mock_trainer_class,
    mock_base_model,
    mock_tokenizer,
    default_finetuning_params
):
    """
    Edge case: If training_data is empty, the code may still try to run.
    Typically, this won't be very useful, but let's see if it handles gracefully.
    """
    mock_trainer = MagicMock()
    mock_trainer.train_dataset = MagicMock()
    mock_trainer.train.return_value = None
    mock_trainer.model = MagicMock()
    mock_trainer_class.return_value = mock_trainer

    empty_data = []

    finetuned_model = finetune_model(
        base_model=mock_base_model,
        tokenizer=mock_tokenizer,
        training_data=empty_data,
        finetuning_params=default_finetuning_params,
        train_lora=False
    )

    # Trainer was still called
    mock_trainer.train.assert_called_once()

    # The returned model is just trainer.model
    assert finetuned_model == mock_trainer.model

def test_finetune_model_no_training_params(
    mock_base_model,
    mock_tokenizer,
    sample_training_data
):
    """
    If finetuning_params is missing, the code uses defaults. 
    This is a smaller test that ensures no KeyErrors occur and training can proceed.
    """
    no_params = {}

    with patch("auto_finetuning_train.Trainer") as mock_trainer_class:
        mock_trainer = MagicMock()
        mock_trainer.train_dataset = MagicMock()
        mock_trainer.train.return_value = None
        mock_trainer.model = MagicMock()
        mock_trainer_class.return_value = mock_trainer

        finetuned_model = finetune_model(
            base_model=mock_base_model,
            tokenizer=mock_tokenizer,
            training_data=sample_training_data,
            finetuning_params=no_params,
            train_lora=False
        )

        mock_trainer.train.assert_called_once()
        assert finetuned_model == mock_trainer.model