from .run_settings import RunSettings, ModelSettings, DataSettings, PromptSettings, PlotSettings, ClusteringSettings, TsneSettings, EmbeddingSettings, DirectorySettings
import yaml
from pathlib import Path
from typing import Dict, List, Any

class RunConfigurationManager:
    def __init__(self):
        self.config_file = Path(__file__).parents[0] / "config.yaml"
        self.configurations: Dict[str, RunSettings] = {}
        self.load_configurations()
        self.run_metadata_file = Path(__file__).parents[3] / "data" / "metadata" / "run_metadata.yaml"

    def load_configurations(self):
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                config_dict = yaml.safe_load(f)
            default_config = config_dict.get('default', {})
            
            # Create a "Default Run" configuration
            default_run_config = default_config.copy()
            default_run_config['name'] = "Default Run"
            self.configurations["Default Run"] = RunSettings.from_dict(default_run_config)
            
            # Process other configurations
            for name, config in config_dict.items():
                if name != 'default':
                    merged_config = self._merge_configs(default_config, config)
                    if 'name' not in merged_config:
                        merged_config['name'] = name
                    self.configurations[name] = RunSettings.from_dict(merged_config)

    def save_configurations(self):
        config_dict = {name: config.to_dict() for name, config in self.configurations.items()}
        with open(self.config_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def get_configuration(self, name: str) -> RunSettings:
        config = self.configurations.get(name)
        if config:
            config.update_n_clusters()
            config.update_tsne_settings()
        return config

    def add_configuration(self, run_settings: RunSettings):
        self.configurations[run_settings.name] = run_settings
        self.save_configurations()

    def remove_configuration(self, name: str):
        if name in self.configurations:
            del self.configurations[name]
            self.save_configurations()

    def list_configurations(self) -> List[str]:
        return list(self.configurations.keys())

    def get_available_sections(self) -> List[str]:
        return [
            "model_comparison",
            "personas_evaluation",
            "awareness_evaluation",
            "hierarchical_clustering"
        ]

    def print_available_sections(self):
        print("Available sections:")
        for section in self.get_available_sections():
            print(f"- {section}")

    @staticmethod
    def _merge_configs(default_config: dict, specific_config: dict) -> dict:
        merged_config = default_config.copy()
        for key, value in specific_config.items():
            if isinstance(value, dict) and key in merged_config:
                merged_config[key] = RunConfigurationManager._merge_configs(merged_config[key], value)
            else:
                merged_config[key] = value
        return merged_config

    def load_run_metadata(self) -> Dict[str, Any]:
        if self.run_metadata_file.exists():
            with open(self.run_metadata_file, 'r') as f:
                try:
                    return yaml.safe_load(f) or {}
                except yaml.YAMLError:
                    print(f"Warning: Could not parse {self.run_metadata_file}. Starting with empty metadata.")
        return {}

    def get_run_metadata(self, run_id: str) -> Dict[str, Any]:
        run_metadata = self.load_run_metadata()
        return run_metadata.get(run_id, {})

    def save_run_metadata(self, run_id: str, metadata: Dict[str, Any]):
        run_metadata = self.load_run_metadata()
        run_metadata[run_id] = self._convert_paths_to_str(metadata)
        with open(self.run_metadata_file, 'w') as f:
            yaml.dump(run_metadata, f, default_flow_style=False)

    @staticmethod
    def _convert_paths_to_str(data):
        if isinstance(data, Path):
            return str(data)
        elif isinstance(data, dict):
            return {k: RunConfigurationManager._convert_paths_to_str(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [RunConfigurationManager._convert_paths_to_str(i) for i in data]
        return data