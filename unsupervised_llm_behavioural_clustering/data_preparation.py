import os
from typing import List, Tuple
import subprocess
import pandas as pd
import numpy as np
import json
from dotenv import load_dotenv


class DataPreparation:
    def __init__(self, file_paths: List[str]):
        self.files_paths = file_paths

    def load_api_key(self, api_key_name: str) -> str:
        """Load API key from a .env file."""
        load_dotenv()  # Load environment variables from a .env file
        api_key = os.getenv(api_key_name)
        if api_key is None:
            raise ValueError(
                f"{api_key_name} not found. Make sure it's stored in the .env file."
            )
        return api_key

    def clone_repo(self, repo_url: str, dest_dir: str) -> None:
        """Clone a GitHub repository."""
        # Using subprocess instead of os.system for better control
        subprocess.run(["git", "clone", repo_url, dest_dir])

    def load_evaluation_data(self) -> List[List[str]]:
        """Load evaluation data from a list of file paths."""
        if not self.file_paths:
            raise ValueError("No file paths provided.")

        all_texts = []
        for path in self.file_paths:
            if not os.path.exists(path):
                continue

            with open(path, "r") as f:
                json_lines = f.readlines()
                dicts = [[path, json.loads(l)] for l in json_lines]
                for d in dicts:
                    if "statement" in d[1]:
                        all_texts.append([d[0], d[1]["statement"]])
        return all_texts
