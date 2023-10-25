import os
import subprocess
import pandas as pd
import numpy as np
import json


def load_api_key(file_path):
    """
    Load API key from a file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"API key file not found at {file_path}")

    with open(file_path, "r") as f:
        api_key = f.read().strip()
    return api_key


def clone_repo(repo_url, dest_dir):
    """
    Clone a GitHub repository.
    """
    # Using subprocess instead of os.system for better control
    subprocess.run(["git", "clone", repo_url, dest_dir])


def load_evaluation_data(file_paths):
    """
    Load evaluation data from a list of file paths.
    """
    # Handle empty list scenario
    if not file_paths:
        raise ValueError("No file paths provided.")

    all_texts = []
    for path in file_paths:
        # Check if file exists
        if not os.path.exists(path):
            continue  # Skip the file or raise an exception based on your requirement

        with open(path, "r") as f:
            json_lines = f.readlines()
            dicts = [[path, json.loads(l)] for l in json_lines]
            for d in dicts:
                if "statement" in d[1]:
                    all_texts.append([d[0], d[1]["statement"]])
    return all_texts
