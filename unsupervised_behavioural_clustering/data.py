# data.py

from typing import List, Tuple
import logging
import glob
import json
import numpy as np
import openai
import pickle

logger = logging.getLogger(__name__)


def load_statements(n_statements: int = 5000) -> List[str]:
    """Load statements from evaluation data JSONL files.

    Args:
        n_statements: Max number of statements to return.

    Returns:
        List of statements.
    """
    statements = []
    file_paths = glob.iglob("evals/**/*.jsonl", recursive=True)
    for path in file_paths:
        try:
            with open(path) as f:
                for line in f:
                    data = json.loads(line)
                    if "statement" in data:
                        statements.append(data["statement"])
        except FileNotFoundError as e:
            logger.error(f"Error loading {path}: {e}")

    rng = np.random.default_rng(seed=42)
    return rng.permutation(statements)[:n_statements]


def load_api_key(key_file: str = "openai_key.txt") -> str:
    """Load OpenAI API key from file."""
    try:
        with open(key_file) as f:
            return f.read()
    except FileNotFoundError as e:
        logger.error(f"Could not load OpenAI key: {e}")
        raise e


def query_and_save_reactions(
    statements: List[str], llms: List, prompt: str, out_file: str
) -> Tuple[List[str], List[str]]:
    """Query LLMs and save results."""
    ...


def test_load_statements():
    """Test statement loading."""
    statements = load_statements(n_statements=10)
    assert len(statements) == 10
