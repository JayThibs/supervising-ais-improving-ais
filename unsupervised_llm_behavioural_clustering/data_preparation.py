import os
import glob
from typing import List, Tuple, Union
import subprocess
import pandas as pd
import numpy as np
import json
import pickle
from dotenv import load_dotenv
from config.run_settings import DataSettings, RunSettings


class DataPreparation:
    def __init__(self):
        self.data_dir = os.getcwd() + "/data"
        self.evals_dir = f"{self.data_dir}/evals"
        print(self.evals_dir)
        self.file_paths = self._get_jsonl_file_paths(self.evals_dir)
        print("Grabbed file paths.")
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

    def load_evaluation_data(self, file_paths: List[str]) -> List[List[str]]:
        """Load evaluation data from a list of file paths."""
        all_texts = []
        for path in file_paths:
            if not os.path.exists(path):
                continue

            with open(path, "r") as f:
                json_lines = f.readlines()
                dicts = [[path, json.loads(l)] for l in json_lines]
                for d in dicts:
                    if "statement" in d[1]:
                        all_texts.append([d[0], d[1]["statement"]])

        print(f"Loaded {len(all_texts)} texts.")
        # Print a few sample elements to check the data structure
        print("Sample elements from all_texts:")
        for i in range(min(3, len(all_texts))):
            print(f"Element {i}: {all_texts[i]}")
        return all_texts

    def load_and_preprocess_data(self, data_settings: DataSettings) -> List[str]:
        """Load and preprocess evaluation data. Return a subset of texts."""
        # Determine file paths based on user input
        if data_settings.datasets == "all":
            file_paths = [
                path for path in glob.iglob("data/evals/**/*.jsonl", recursive=True)
            ]
        elif isinstance(data_settings.datasets, list):
            file_paths = []
            for dataset in data_settings.datasets:
                file_paths.extend(
                    [
                        path
                        for path in glob.iglob(
                            f"data/evals/**/{dataset}.jsonl", recursive=True
                        )
                    ]
                )
        else:  # Single dataset specified
            file_paths = [
                path
                for path in glob.iglob(
                    f"data/evals/**/{data_settings.datasets}.jsonl", recursive=True
                )
            ]

        print(f"Found {len(file_paths)} files.")
        all_texts = self.load_evaluation_data(file_paths)
        short_texts = self.load_short_texts(all_texts)
        print(f"Sample short texts: {short_texts[:2]}")
        text_subset = self.create_text_subset(short_texts, data_settings.n_statements)
        print(f"Sample text subset: {text_subset[:2]}")
        print(f"Loaded {len(all_texts)} texts.")
        print(f"Loaded {len(short_texts)} short texts.")
        print(f"Loaded {len(text_subset)} text subset.")
        print("Sample elements from text_subset:")
        for i in range(min(3, len(text_subset))):
            print(f"Element {i}: {text_subset[i]}")
        return text_subset  # numpy.ndarray

    def load_short_texts(self, all_texts, max_length=150):
        return [t[1] for t in all_texts if len(t[1]) < max_length]

    def create_text_subset(self, texts, data_settings: DataSettings):
        rng = np.random.default_rng(seed=data_settings.random_state)
        return rng.permutation(texts)[: data_settings.n_statements]

    def save_to_pickle(self, data, filename, run_settings: RunSettings):
        pickle_dir = f"{run_settings.results_dir}/pickle_files"
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir)
        with open(os.path.join(pickle_dir, filename), "wb") as f:
            pickle.dump(data, f)

    def load_from_pickle(self, filename):
        pickle_dir = "data/intermediate/pickle_files"
        with open(os.path.join(pickle_dir, filename), "rb") as f:
            return pickle.load(f)
