import os
import glob
from typing import List, Tuple
import subprocess
import pandas as pd
import numpy as np
import json
import pickle
from dotenv import load_dotenv


class DataPreparation:
    def __init__(self):
        self.data_dir = os.getcwd() + "/data"
        self.evals_dir = f"{self.data_dir}/evals"
        print(self.evals_dir)
        self.file_paths = self._get_jsonl_file_paths(self.evals_dir)
        print(self.file_paths)
        self.folder_path = self.evals_dir

    @staticmethod
    def load_api_key(self, api_key_name: str) -> str:
        """Load API key from a .env file."""
        load_dotenv()  # Load environment variables from a .env file
        api_key = os.getenv(api_key_name)
        if api_key is None:
            raise ValueError(
                f"{api_key_name} not found. Make sure it's stored in the .env file."
            )
        return api_key

    @staticmethod
    def clone_repo(self, repo_url: str, folder_name: str) -> None:
        """Clone a GitHub repository."""
        # Using subprocess instead of os.system for better control
        subprocess.run(
            ["git", "clone", repo_url, os.path.join(self.evals_dir, folder_name)]
        )

    def _get_subfolders(self, folder_path: str = None) -> List[str]:
        """Find all subfolders within the specified folder."""
        if folder_path is None:
            folder_path = self.evals_dir
        return [f.name for f in os.scandir(self.evals_dir) if f.is_dir()]

    def _get_jsonl_file_paths(self, folder_path: str = None) -> List[str]:
        """Find all .jsonl files within the specified folder and subfolders."""
        subfolders = self._get_subfolders()
        file_paths = []
        for subfolder in subfolders:
            file_paths.extend(
                glob.glob(f"{self.evals_dir}/{subfolder}/**/**/*.jsonl", recursive=True)
            )
        return file_paths

    def load_evaluation_data(self) -> List[List[str]]:
        """Load evaluation data from a list of file paths."""
        all_texts = []
        for path in self.file_paths:
            if not os.path.exists(path):
                continue

            with open(path, "r") as f:
                json_lines = f.readlines()
                dicts = [[path, json.loads(l)] for l in json_lines]
                all_texts.extend([d["statements"] for d in dicts if "statement" in d])

        return all_texts

    def load_short_texts(self, all_texts, max_length=150):
        return [t[1] for t in all_texts if len(t[1]) < max_length]

    def create_text_subset(self, texts, n_points=5000, seed=42):
        rng = np.random.default_rng(seed=seed)
        return rng.permutation(texts)[:n_points]

    def save_to_pickle(self, data, filename):
        with open(filename, "wb") as f:
            pickle.dump(data, f)

    def load_from_pickle(self, filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
