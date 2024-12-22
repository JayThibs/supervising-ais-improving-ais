# test_auto_finetuning_helpers.py

import os
import pytest
import json
from unittest.mock import patch, MagicMock, mock_open
import sys
sys.path.append("../")
from auto_finetuning_helpers import (
    load_api_key,
    make_api_request,
    parallel_make_api_requests,
    extract_json_from_string,
    # Add any other methods if you want to test them
)

# ---------------------------------------------------------------------
# 1. load_api_key Tests
# ---------------------------------------------------------------------

def test_load_api_key_no_provider(tmp_path):
    """
    Ensures load_api_key works when only a single key is in the file and no provider is specified.
    """
    key_file = tmp_path / "key.txt"
    key_file.write_text("my_single_api_key_12345")

    result = load_api_key(str(key_file))
    assert result == "my_single_api_key_12345", "Should load the entire file content as the key"


def test_load_api_key_with_provider(tmp_path):
    """
    Tests that load_api_key can parse multiple keys from a single file if lines are in the form 'provider:key'
    """
    key_file = tmp_path / "key.txt"
    key_file.write_text(
        "anthropic:anthropic_key_abc123\n"
        "openai:openai_key_def456\n"
    )

    result_anthropic = load_api_key(str(key_file), api_provider="anthropic")
    assert result_anthropic == "anthropic_key_abc123"

    result_openai = load_api_key(str(key_file), api_provider="openai")
    assert result_openai == "openai_key_def456"


def test_load_api_key_missing_file():
    """
    If the file doesn't exist, we expect a FileNotFoundError.
    """
    with pytest.raises(FileNotFoundError):
        load_api_key("non_existent_file.txt")


def test_load_api_key_missing_provider(tmp_path):
    """
    If we ask for a provider that doesn't exist in the file, we expect ValueError.
    """
    key_file = tmp_path / "key.txt"
    key_file.write_text("anthropic:anthropic_key_abc123")

    with pytest.raises(ValueError):
        load_api_key(str(key_file), api_provider="openai")


# ---------------------------------------------------------------------
# 2. extract_json_from_string Tests
# ---------------------------------------------------------------------

def test_extract_json_from_string_valid():
    response = """Here is some text:
    [
        "Item 1",
        "Item 2"
    ]
    Extra text at the end.
    """
    extracted = extract_json_from_string(response)
    assert extracted == ["Item 1", "Item 2"], "Should extract JSON array from text"


def test_extract_json_from_string_dict():
    response = """Random prefix
    {
      "status": "success",
      "value": 42
    }
    Some other text
    """
    extracted = extract_json_from_string(response)
    assert extracted["status"] == "success"
    assert extracted["value"] == 42


def test_extract_json_from_string_no_json():
    response = "No brackets at all here"
    with pytest.raises(ValueError):
        extract_json_from_string(response)


# def test_extract_json_from_string_malformed():
#     response = "Some text [ not valid JSON"
#     # Depending on how your code is structured, it might either raise or return []
#     # We'll assume it prints an error and returns []
#     result = extract_json_from_string(response)
#     assert result == [], "Should return empty list on final parsing error"


# ---------------------------------------------------------------------
# 3. make_api_request Tests (mock-based)
# ---------------------------------------------------------------------

@patch("auto_finetuning_helpers.time.sleep", return_value=None)
def test_make_api_request_retries(mock_sleep):
    """
    If the API call fails, it should retry up to n_local_retries.
    We'll patch the function that actually calls the API and force an exception to test retries.
    """
    with patch("auto_finetuning_helpers.Anthropic") as mock_anthropic:
        # Mock the client so that it raises an error on first calls, then succeeds
        mock_client_instance = MagicMock()
        # Raise an exception 2 times, then return a success on the 3rd
        mock_client_instance.messages.create.side_effect = [
            Exception("InternalServerError mock"),
            Exception("InternalServerError mock"),
            MagicMock(content=[MagicMock(text="Hello world!")]),
        ]
        mock_anthropic.return_value = mock_client_instance

        result = make_api_request(
            prompt="Some test prompt",
            api_provider="anthropic",
            model_str="fake-model",
            api_key="fake-key",
            n_local_retries=3
        )
        assert "Hello world!" in result, "Expected the final (3rd) call to succeed"
        # The time.sleep should be called at least twice
        assert mock_sleep.call_count >= 2, "Expected multiple retries"


# ---------------------------------------------------------------------
# 4. parallel_make_api_requests Tests
# ---------------------------------------------------------------------

@patch("auto_finetuning_helpers.make_api_request")
def test_parallel_make_api_requests_mocked(mock_make_api_request):
    """
    Unit test that checks the logic of parallel_make_api_requests when calls are mocked.
    """
    # We'll simulate each prompt returning "MOCKED_RESPONSE"
    mock_make_api_request.return_value = "MOCKED_RESPONSE"

    prompts = ["prompt1", "prompt2", "prompt3"]
    results = parallel_make_api_requests(
        prompts,
        api_provider="anthropic",
        api_model_str="fake-model",
        auth_key="some_key"
    )
    assert len(results) == len(prompts)
    assert all(r == "MOCKED_RESPONSE" for r in results), "All results should be 'MOCKED_RESPONSE'"
    # Check that the underlying make_api_request was called exactly len(prompts) times
    assert mock_make_api_request.call_count == len(prompts)


@pytest.mark.skipif(
    "ANTHROPIC_KEY" not in os.environ and "OPENAI_KEY" not in os.environ,
    reason="No real API key found in environment to run real parallel_make_api_requests test"
)
def test_parallel_make_api_requests_real():
    """
    Basic integration-style test that calls an actual API.
    Make sure you have ANTHROPIC_KEY or OPENAI_KEY in the environment.
    
    This won't do heavy logic checks beyond seeing if we get responses back without errors.
    """
    # Decide which provider to test. We'll pick OpenAI if OPENAI_KEY is set, otherwise Anthropic
    if "OPENAI_KEY" in os.environ:
        provider = "openai"
        key = os.environ["OPENAI_KEY"]
        model_str = "gpt-3.5-turbo"
    else:
        provider = "anthropic"
        key = os.environ["ANTHROPIC_KEY"]
        model_str = "claude-instant-1"

    prompts = [
        "Hello, can you briefly say how you differ from a standard chatbot?",
        "What's the capital of France?",
    ]

    # Attempt the real call
    responses = parallel_make_api_requests(
        prompts,
        api_provider=provider,
        api_model_str=model_str,
        auth_key=key,
        num_workers=2,
        max_tokens=100
    )

    # We won't over-assert on the content, just ensure we got a well-formed response
    assert len(responses) == len(prompts)
    for resp in responses:
        assert isinstance(resp, str) and len(resp) > 0, "Each response should be a non-empty string"