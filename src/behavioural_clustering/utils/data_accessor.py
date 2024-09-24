import yaml
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

class DataAccessor:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.metadata_path = self.base_dir / "metadata" / "data_metadata.yaml"
        self.run_metadata_path = self.base_dir / "metadata" / "run_metadata.yaml"
        self.load_metadata()

    def load_metadata(self):
        with open(self.metadata_path, 'r') as f:
            self.data_metadata = yaml.safe_load(f)
        with open(self.run_metadata_path, 'r') as f:
            self.run_metadata = yaml.safe_load(f)

    def get_run_data(self, run_id, data_type):
        if run_id not in self.run_metadata:
            raise ValueError(f"Run ID {run_id} not found in metadata")
        
        file_id = self.run_metadata[run_id]['data_file_ids'].get(data_type)
        if not file_id:
            raise ValueError(f"Data type {data_type} not found for run {run_id}")
        
        # Construct the full file path
        file_path = self.base_dir / "saved_data" / data_type / f"{file_id}.pkl"
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        data = self.load_data_file(file_path)
        
        # Debug information
        print(f"Debug - Run {run_id} - {data_type}:")
        print(f"  Type: {type(data)}")
        print(f"  Length/Shape: {len(data) if isinstance(data, list) else data.shape if isinstance(data, np.ndarray) else 'N/A'}")
        if isinstance(data, list) and len(data) > 0:
            print(f"  First item type: {type(data[0])}")
            if isinstance(data[0], dict):
                print(f"  First item keys: {data[0].keys()}")
                if 'model_name' in data[0]:
                    print(f"  Model names in data: {set(item['model_name'] for item in data)}")
        
        # Add more specific debug information for approvals data
        if data_type.startswith("approvals_statements_"):
            print(f"Debug - Run {run_id} - {data_type}:")
            print(f"  Type: {type(data)}")
            print(f"  Length: {len(data)}")
            if isinstance(data, list) and len(data) > 0:
                print(f"  First item type: {type(data[0])}")
                if isinstance(data[0], dict):
                    print(f"  First item keys: {data[0].keys()}")
                    if 'approvals' in data[0]:
                        print(f"  Models in approvals: {set(model for item in data for model in item['approvals'].keys())}")
                        print(f"  Prompt types in approvals: {set(prompt_type for item in data for model_approvals in item['approvals'].values() for prompt_type in model_approvals.keys())}")
        
        # Data validation and transformation
        if data_type == "joint_embeddings_all_llms":
            # Convert loaded embeddings back to numpy arrays if necessary
            if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                data = [{**e, "embedding": np.array(e["embedding"])} for e in data]
        elif data_type == "spectral_clustering":
            if hasattr(data, 'labels_'):
                data = data.labels_
            else:
                raise ValueError(f"Invalid spectral clustering data for run {run_id}. Expected SpectralClustering object with labels_.")
        
        return data

    def load_data_file(self, file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def get_run_config(self, run_id: str):
        return self.run_metadata[run_id]

    def list_runs(self):
        return list(self.run_metadata.keys())

    def list_data_types(self, run_id: str):
        return list(self.run_metadata[run_id]["data_file_ids"].keys())

    def get_available_run_ids(self):
        return list(self.run_metadata.keys())

    def get_available_prompt_types(self, run_id):
        if run_id not in self.run_metadata:
            raise ValueError(f"Run ID {run_id} not found in run metadata.")
        return [key for key in self.run_metadata[run_id]["data_file_ids"].keys() if key.startswith("approvals_statements_")]

    def get_file_path(self, run_id: str, data_type: str) -> Path:
        if run_id not in self.run_metadata:
            raise ValueError(f"Run ID {run_id} not found in run metadata.")
        
        # Handle the case where data_type is 'compile_cluster_table'
        if data_type == 'compile_cluster_table':
            file_id = self.run_metadata[run_id]["data_file_ids"].get(data_type)
        else:
            # Split the data_type if it contains the run_id
            _, data_type = data_type.rsplit('_', 1) if '_' in data_type else ('', data_type)
            file_id = self.run_metadata[run_id]["data_file_ids"].get(data_type)
        
        if not file_id:
            raise ValueError(f"Data type {data_type} not found for run {run_id}.")
        
        return self.base_dir / "saved_data" / data_type / f"{file_id}.pkl"

    def get_dataset_names(self, run_id: str) -> str:
        if run_id not in self.run_metadata:
            raise ValueError(f"Run ID {run_id} not found in run metadata.")
        
        dataset_names = self.run_metadata[run_id].get("dataset_names", [])
        if isinstance(dataset_names, list):
            return ", ".join(dataset_names)
        elif isinstance(dataset_names, str):
            return dataset_names
        else:
            return str(dataset_names)

    def get_prompt_categories(self, run_id: str):
        run_config = self.get_run_config(run_id)
        return run_config.get("prompt_categories", [])

    def get_prompt_labels(self, run_id: str, category: str):
        run_config = self.get_run_config(run_id)
        prompts_file = self.base_dir / "prompts" / "approval_prompts.json"
        with open(prompts_file, 'r') as f:
            prompts_data = json.load(f)
        return list(prompts_data.get(category, {}).keys())

    def debug_pickle_file(self, run_id, data_type):
        file_id = self.run_metadata[run_id]['data_file_ids'].get(data_type)
        file_path = self.base_dir / "saved_data" / data_type / f"{file_id}.pkl"
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Debug {run_id} - {data_type}:")
        print(f"  Type: {type(data)}")
        print(f"  Content: {data}")

    def get_model_names(self, run_id: str) -> list:
        if run_id not in self.run_metadata:
            raise ValueError(f"Run ID {run_id} not found in run metadata.")
        
        return self.run_metadata[run_id].get("model_names", [])

    def get_visualization_directory(self, run_id: str) -> str:
        if run_id not in self.run_metadata:
            raise ValueError(f"Run ID {run_id} not found in run metadata.")
        
        viz_dir = self.run_metadata[run_id].get("visualization_directory", "")
        if not viz_dir:
            # If no visualization directory is specified, create one based on the run_id
            viz_dir = str(self.base_dir / "visualizations" / run_id)
        
        return viz_dir