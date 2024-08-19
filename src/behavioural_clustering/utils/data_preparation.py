import os
import glob
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
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
import uuid
import hashlib
import traceback


class DataPreparation:
    def __init__(self):
        self.data_dir = Path.cwd() / "data"
        self.evals_dir = self.data_dir / "evals"
        self.pickle_dir = self.data_dir / "results" / "pickle_files"
        self.log_dir = self.data_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)
        self.accepted_file = self.log_dir / "accepted_jsonl.txt"
        self.rejected_file = self.log_dir / "rejected_jsonl.txt"
        print(f"Evals directory: {self.evals_dir}")
        self.file_paths = self._get_jsonl_file_paths()
        print("Grabbed file paths.")

    @staticmethod
    def load_api_key(api_key_name: str) -> str:
        """Load API key from a .env file."""
        load_dotenv()  # Load environment variables from a .env file
        api_key = os.getenv(api_key_name)
        if api_key is None:
            raise ValueError(
                f"{api_key_name} not found. Make sure it's stored in the .env file."
            )
        return api_key

    @staticmethod
    def clone_repo(repo_url: str, evals_dir: Path, folder_name: str) -> None:
        """Clone a GitHub repository."""
        subprocess.run(["git", "clone", repo_url, evals_dir / folder_name])

    def _get_subfolders(self) -> List[str]:
        """Find all subfolders within the evals directory."""
        return [f.name for f in self.evals_dir.iterdir() if f.is_dir()]

    def _get_jsonl_file_paths(self) -> List[str]:
        """Find all .jsonl files within the evals directory and subfolders."""
        subfolders = self._get_subfolders()
        file_paths = []
        for subfolder in subfolders:
            file_paths.extend(
                glob.glob(str(self.evals_dir / subfolder / "**" / "*.jsonl"), recursive=True)
            )
        return file_paths

    def load_evaluation_data(self, datasets: List[str]) -> List[Tuple[str, str]]:
        """Load evaluation data from the specified datasets."""
        all_texts = []
        accepted_count = 0
        rejected_count = 0
        print(f"Attempting to load datasets: {datasets}")
        
        with open(self.accepted_file, 'w') as accepted_f, open(self.rejected_file, 'w') as rejected_f:
            for dataset in datasets:
                dataset_path = self.evals_dir / dataset
                file_paths = glob.glob(str(dataset_path / "**" / "*.jsonl"), recursive=True)
                print(f"Found {len(file_paths)} JSONL files in {dataset}")
                for path in file_paths:
                    texts, accepted = self._load_jsonl_file(Path(path))
                    all_texts.extend(texts)
                    if accepted:
                        accepted_f.write(f"{path}\n")
                        accepted_count += 1
                    else:
                        rejected_f.write(f"{path}\n")
                        rejected_count += 1

        print(f"Total loaded texts: {len(all_texts)}")
        print(f"Accepted JSONL files: {accepted_count}")
        print(f"Rejected JSONL files: {rejected_count}")
        return all_texts

    def _load_jsonl_file(self, file_path: Path) -> Tuple[List[Tuple[str, str]], bool]:
        texts = []
        accepted = False
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if "statement" in data:
                                texts.append((str(file_path), data["statement"]))
                                accepted = True
                            elif "question" in data:
                                texts.append((str(file_path), data["question"]))
                                accepted = True
                        except json.JSONDecodeError:
                            pass  # Silently ignore invalid JSON
        except Exception as e:
            pass  # Silently ignore file read errors

        return texts, accepted

    def load_short_texts(self, all_texts, max_length=150):
        return [t[1] for t in all_texts if len(t[1]) < max_length]

    def create_text_subset(self, texts, data_settings: DataSettings):
        if not texts:
            return []
        rng = np.random.default_rng(seed=data_settings.random_state)
        return rng.permutation(texts)[:data_settings.n_statements].tolist()

    def load_and_preprocess_data(self, data_settings: DataSettings) -> List[str]:
        """Load and preprocess evaluation data. Return a subset of texts."""
        all_texts = self.load_evaluation_data(data_settings.datasets)
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

    def save_to_pickle(self, data, filename, run_settings: RunSettings):
        pickle_dir = run_settings.directory_settings.pickle_dir
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir)
        with open(pickle_dir / filename, "wb") as f:
            pickle.dump(data, f)

    def load_from_pickle(self, filename, run_settings: RunSettings):
        pickle_dir = run_settings.directory_settings.pickle_dir
        with open(pickle_dir / filename, "rb") as f:
            return pickle.load(f)

    def load_pkl_or_not(
        self, filename: str, directory: str, load_if_exists: bool
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
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_filename = f"{filename[:-4]}_{timestamp}.pkl"
                new_filepath = os.path.join(directory, new_filename)
                os.rename(filepath, new_filepath)
                logging.info(f"Saved old {filename} as {new_filename}.")

            return False, None


class DataHandler:
    def __init__(self, base_dir: Path, run_id: str):
        self.base_dir = base_dir
        self.data_dir = base_dir / "saved_data"
        self.metadata_dir = base_dir / "metadata"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.run_metadata_file = self.metadata_dir / "run_metadata.yaml"
        self.data_metadata_file = self.metadata_dir / "data_metadata.yaml"
        self.run_id = run_id
        self.run_metadata = self.load_run_metadata()
        self.data_metadata = self.load_data_metadata()

    def load_run_metadata(self) -> Dict[str, Any]:
        if self.run_metadata_file.exists():
            with open(self.run_metadata_file, 'r') as f:
                try:
                    all_metadata = yaml.safe_load(f) or {}
                    return all_metadata.get(self.run_id, {})
                except yaml.YAMLError:
                    print(f"Warning: Could not parse {self.run_metadata_file}. Starting with empty metadata.")
        return {}

    def load_data_metadata(self) -> Dict[str, Any]:
        if self.data_metadata_file.exists():
            with open(self.data_metadata_file, 'r') as f:
                try:
                    return yaml.safe_load(f) or {}
                except yaml.YAMLError:
                    print(f"Warning: Could not parse {self.data_metadata_file}. Starting with empty metadata.")
        return {}

    def save_run_metadata(self, metadata: Dict[str, Any]):
        all_metadata = {}
        if self.run_metadata_file.exists():
            with open(self.run_metadata_file, 'r') as f:
                all_metadata = yaml.safe_load(f) or {}
        all_metadata[self.run_id] = metadata
        with open(self.run_metadata_file, 'w') as f:
            yaml.dump(all_metadata, f, default_flow_style=False)

    def save_data_metadata(self):
        with open(self.data_metadata_file, 'w') as f:
            yaml.dump(self.data_metadata, f, default_flow_style=False)

    def save_data(self, data: Any, data_type: str, config: Dict[str, Any]) -> str:
        file_id = self._generate_file_id(data_type, config)
        data_type_dir = self.data_dir / data_type
        data_type_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"{file_id}.pkl"
        file_path = data_type_dir / file_name
        
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

        if data_type not in self.data_metadata:
            self.data_metadata[data_type] = {}
        self.data_metadata[data_type][file_id] = {
            "file_path": str(file_path),
            "config": config
        }
        self.save_data_metadata()
        
        print(f"Saved {data_type} to file: {file_path}")
        return file_id

    def _generate_file_id(self, data_type: str, config: Dict[str, Any]) -> str:
        relevant_config = self._get_relevant_config(data_type, config)
        config_str = json.dumps(relevant_config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def _get_relevant_config(self, data_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        if data_type == "all_query_results":
            return {
                "datasets": config["data_settings"]["datasets"],
                "n_statements": config["data_settings"]["n_statements"],
                "random_state": config["data_settings"]["random_state"],
                "model_settings": config["model_settings"],
                "statements_prompt_template": config["prompt_settings"]["statements_prompt_template"]
            }
        elif data_type in ["joint_embeddings_all_llms", "combined_embeddings"]:
            return {
                "datasets": config["data_settings"]["datasets"],
                "n_statements": config["data_settings"]["n_statements"],
                "random_state": config["data_settings"]["random_state"],
                "model_settings": config["model_settings"],
                "statements_prompt_template": config["prompt_settings"]["statements_prompt_template"],
                "embedding_settings": config["embedding_settings"]
            }
        elif data_type == "chosen_clustering":
            return {
                "datasets": config["data_settings"]["datasets"],
                "n_statements": config["data_settings"]["n_statements"],
                "random_state": config["data_settings"]["random_state"],
                "model_settings": config["model_settings"],
                "statements_prompt_template": config["prompt_settings"]["statements_prompt_template"],
                "embedding_settings": config["embedding_settings"],
                "clustering_settings": config["clustering_settings"]
            }
        else:
            return config

    def load_data(self, data_type: str, config: Dict[str, Any]) -> Optional[Any]:
        file_id = self._generate_file_id(data_type, config)
        if data_type in self.data_metadata and file_id in self.data_metadata[data_type]:
            file_path = self.data_metadata[data_type][file_id]["file_path"]
            return self._load_file(file_path)
        print(f"No matching {data_type} found")
        return None

    def _load_file(self, file_path: str) -> Optional[Any]:
        file_path = Path(file_path)
        if file_path.exists():
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                print(f"Loaded data from file: {file_path}")
                return data
            except Exception as e:
                print(f"Error loading data from {file_path}: {str(e)}")
                print("Traceback:")
                traceback.print_exc()
        else:
            print(f"File not found: {file_path}")
        return None

    def list_available_data(self) -> Dict[str, List[str]]:
        available_data = {}
        for data_type, file_ids in self.data_metadata.items():
            available_data[data_type] = list(file_ids.keys())
        return available_data

    def validate_data(self, data_type: str, data: Any) -> bool:
        """
        Validate the loaded data based on its expected type and structure.
        """
        if not isinstance(data, (list, dict)):
            return False
        
        if data_type == "text_subset":
            return isinstance(data, list) and all(isinstance(item, str) for item in data)
        elif data_type in ["joint_embeddings", "combined_embeddings"]:
            return isinstance(data, dict) and all(isinstance(v, np.ndarray) for v in data.values())
        elif data_type == "chosen_clustering":
            return isinstance(data, (np.ndarray, list))
        elif data_type == "rows":
            return isinstance(data, list) and all(isinstance(item, dict) for item in data)
        elif data_type.startswith("approvals_") or data_type.startswith("hierarchy_data_"):
            return isinstance(data, dict)
        else:
            # For unknown data types, we'll assume it's valid as long as it's a list or dict
            return True

    def find_matching_files(self, data_type: str, metadata: dict) -> List[Path]:
        matching_files = []
        for file_type, file_path in self.metadata.get('data_files', {}).items():
            if file_type == data_type:
                # Check if all metadata items match
                if all(self.metadata.get(k) == v for k, v in metadata.items()):
                    file_path = Path(file_path)
                    if file_path.exists():
                        matching_files.append(file_path)
        return matching_files

    def list_files_with_metadata(self, data_type: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        matching_files = []
        for data_type, file_ids in self.data_metadata.items():
            for file_id, metadata in file_ids.items():
                if (data_type is None or data_type == metadata['data_type']) and metadata['metadata'] == kwargs:
                    matching_files.append({
                        "file_id": file_id,
                        "file_path": metadata['file_path'],
                        "data_type": metadata['data_type'],
                        "metadata": metadata['metadata']
                    })
        return matching_files