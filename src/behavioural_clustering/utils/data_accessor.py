from pathlib import Path
import yaml
import pickle
import json

class DataAccessor:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.run_metadata_file = base_dir / "metadata" / "run_metadata.yaml"
        self.data_metadata_file = base_dir / "metadata" / "data_metadata.yaml"
        self.run_metadata = self.load_yaml(self.run_metadata_file)
        self.data_metadata = self.load_yaml(self.data_metadata_file)

    def load_yaml(self, file_path: Path):
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
            return self._convert_str_to_paths(data)

    @staticmethod
    def _convert_str_to_paths(data):
        if isinstance(data, str) and ('/' in data or '\\' in data):
            return Path(data)
        elif isinstance(data, dict):
            return {k: DataAccessor._convert_str_to_paths(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [DataAccessor._convert_str_to_paths(i) for i in data]
        return data

    def get_run_data(self, run_id: str, data_type: str):
        if run_id not in self.run_metadata:
            raise ValueError(f"Run ID {run_id} not found in run metadata.")
        
        file_id = self.run_metadata[run_id]["data_file_ids"].get(data_type)
        if not file_id:
            raise ValueError(f"Data type {data_type} not found for run {run_id}.")
        
        file_path = self.data_metadata[data_type][file_id]["file_path"]
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
        
        file_id = self.run_metadata[run_id]["data_file_ids"].get(data_type)
        if not file_id:
            raise ValueError(f"Data type {data_type} not found for run {run_id}.")
        
        file_path = self.data_metadata[data_type][file_id]["file_path"]
        return Path(file_path)

    def get_prompt_categories(self, run_id: str):
        run_config = self.get_run_config(run_id)
        return run_config.get("prompt_categories", [])

    def get_prompt_labels(self, run_id: str, category: str):
        run_config = self.get_run_config(run_id)
        prompts_file = self.base_dir / "prompts" / "approval_prompts.json"
        with open(prompts_file, 'r') as f:
            prompts_data = json.load(f)
        return list(prompts_data.get(category, {}).keys())