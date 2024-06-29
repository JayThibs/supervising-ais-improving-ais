import os
from pathlib import Path
from typing import List, Tuple, Union, Dict, Any, Optional
import logging
import subprocess
from threading import Lock
import numpy as np
import json
import yaml
import pickle
from datetime import datetime
from dotenv import load_dotenv
from behavioural_clustering.config.run_settings import DataSettings, RunSettings


class DataPreparation:
    def __init__(self):
        self.data_dir = Path.cwd() / "data"
        self.evals_dir = self.data_dir / "evals"
        self.file_paths = self._get_jsonl_file_paths(self.evals_dir)
        print("Grabbed file paths.")

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
        subprocess.run(["git", "clone", repo_url, self.evals_dir / folder_name])

    def _get_jsonl_file_paths(
        self, folder_path: Path = None, datasets: Union[List[str], str, None] = None
    ) -> List[str]:
        """Find all .jsonl files within the specified folder and subfolders, filtered by datasets if provided."""
        if folder_path is None:
            folder_path = self.evals_dir
        file_paths = []
        if datasets is None:
            datasets = ["all"]
        if datasets == ["all"]:
            file_paths = [str(path) for path in folder_path.rglob("*.jsonl")]
        elif isinstance(datasets, list):
            for dataset in datasets:
                file_paths.extend(
                    [
                        str(path)
                        for path in folder_path.joinpath(dataset).rglob("*.jsonl")
                    ]
                )
        else:  # Single dataset specified
            file_paths = [
                str(path) for path in folder_path.joinpath(datasets).rglob("*.jsonl")
            ]
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
        file_paths = self._get_jsonl_file_paths(datasets=data_settings.datasets)
        print(f"Found {len(file_paths)} files.")
        all_texts = self.load_evaluation_data(file_paths)
        short_texts = self.load_short_texts(all_texts)
        print(f"Sample short texts: {short_texts[:2]}")
        text_subset = self.create_text_subset(short_texts, data_settings)
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
        pickle_dir = run_settings.directory_settings.pickle_dir
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir)
        with open(pickle_dir / filename, "wb") as f:
            pickle.dump(data, f)

    def load_from_pickle(self, filename):
        pickle_dir = self.run_settings.directory_settings.pickle_dir
        with open(pickle_dir / filename, "rb") as f:
            return pickle.load(f)

    def load_pkl_or_not(
        filename: str, directory: str, load_if_exists: bool
    ) -> Tuple[bool, Optional[Any]]:
        """
        Loads a file if it exists and the condition allows, or prepares for the creation
        of a new file by renaming the existing one.

        Args:
        - filename (str): Name of the file.
        - directory (str): Directory where the file is stored.
        - load_if_exists (bool): If True, load file if it exists. If False, rename the old file with a timestamp.

        Returns:
        - Tuple[bool, Optional[Any]]: A tuple where the first element is a boolean indicating if the file was loaded.
                                    The second element is the loaded content or None.
        """
        filepath = os.path.join(directory, filename)

        if load_if_exists and os.path.exists(filepath):
            logging.info(f"Loading {filename}...")
            with open(filepath, "rb") as file:
                return True, pickle.load(file)
        else:
            if not load_if_exists and os.path.exists(filepath):
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                new_filename = f"{filename[:-4]}_{timestamp}.pkl"
                new_filepath = os.path.join(directory, new_filename)
                os.rename(filepath, new_filepath)
                logging.info(f"Saved old {filename} as {new_filename}.")

            return False, None



class DataHandler:
    def __init__(self, results_dir: Path, pickle_dir: Path, data_file_mapping: Path):
        self.results_dir = results_dir
        self.pickle_dir = pickle_dir
        self.run_id = self.generate_run_id()
        self.data_file_mapping = data_file_mapping
        if not self.data_file_mapping.exists():
            with open(self.data_file_mapping, "w") as f:
                yaml.safe_dump({}, f)

    def generate_run_id(self) -> str:
        metadata_file = self.results_dir / "run_metadata.yaml"
        if metadata_file.exists():
            with open(metadata_file, "r") as file:
                existing_metadata = yaml.safe_load(file) or {}
            # Extract run IDs and find the maximum
            last_run_id = max(
                (int(run_id.split("_")[1]) for run_id in existing_metadata.keys()),
                default=0,
            )
        else:
            last_run_id = 0
        new_run_id = f"run_{last_run_id + 1}"
        return new_run_id

    def save_data(self, data: Any, filename: str):
        path = self.pickle_dir / filename
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def save_pickles(
        self,
        data_file_paths: Dict[str, str],
        data: List[Any],
    ):
        for key, path in data_file_paths.items():
            self.save_data(data[key], path)
            print(f"Saved data to {path}.")

    def save_run_metadata_to_yaml(self, run_id: str, metadata: Dict[str, Any]):
        metadata_file = self.results_dir / "run_metadata.yaml"
        lock = Lock()
        with lock:
            if metadata_file.exists():
                with open(metadata_file, "r") as file:
                    existing_metadata = yaml.safe_load(file) or {}
            else:
                existing_metadata = {}

            existing_metadata[run_id] = metadata

            with open(metadata_file, "w") as file:
                yaml.safe_dump(existing_metadata, file)

    def get_or_create_data_file_path(
        self, data_type: str, data_file_dir: Path, mapping_file: Path, **kwargs
    ) -> Path:
        with open(mapping_file, "r") as f:
            try:
                data_file_mapping = yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                data_file_mapping = {}

        # Generate the filename based on the provided arguments
        filename_parts = [data_type]
        for key, value in kwargs.items():
            filename_parts.append(f"{key}_{value}")
        filename = "_".join(filename_parts) + ".pkl"

        if filename in data_file_mapping:
            return Path(data_file_mapping[filename])
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file_path = str(data_file_dir / f"{filename}_{timestamp}.pkl")
            data_file_mapping[filename] = file_path

            with open(mapping_file, "w") as f:
                yaml.safe_dump(data_file_mapping, f)

            return Path(file_path)

    def load_data(self, filename: str) -> Any:
        path = self.pickle_dir / filename
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    def get_run_metadata(self, run_id: str) -> Dict[str, Any]:
        metadata_file = self.results_dir / "run_metadata.yaml"
        with open(metadata_file, "r") as file:
            metadata = yaml.safe_load(file)
        return metadata.get(run_id, {})
