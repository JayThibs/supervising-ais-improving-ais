from .run_settings import RunSettings, ModelSettings, DataSettings, PromptSettings, PlotSettings, ClusteringSettings, TsneSettings, EmbeddingSettings
import yaml
from pathlib import Path
from typing import Dict, List

class RunConfigurationManager:
    def __init__(self):
        self.config_file = Path(__file__).parent / "config.yaml"
        self.configurations: Dict[str, RunSettings] = {}
        self.load_configurations()

    def load_configurations(self):
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                config_dict = yaml.safe_load(f)
            default_config = config_dict.pop('default', {})
            for name, config in config_dict.items():
                self.configurations[name] = self._dict_to_run_settings(config, default_config)

    def save_configurations(self):
        config_dict = {name: self._run_settings_to_dict(config) for name, config in self.configurations.items()}
        with open(self.config_file, 'w') as f:
            yaml.dump(config_dict, f)

    def get_configuration(self, name: str) -> RunSettings:
        config = self.configurations.get(name)
        if config:
            config.update_n_clusters()
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
    def _run_settings_to_dict(run_settings: RunSettings) -> dict:
        return {
            'name': run_settings.name,
            'random_state': run_settings.random_state,
            'model_settings': run_settings.model_settings.__dict__,
            'embedding_settings': run_settings.embedding_settings.__dict__,
            'data_settings': run_settings.data_settings.__dict__,
            'prompt_settings': run_settings.prompt_settings.__dict__,
            'plot_settings': run_settings.plot_settings.__dict__,
            'clustering_settings': run_settings.clustering_settings.__dict__,
            'tsne_settings': run_settings.tsne_settings.__dict__,
            'test_mode': run_settings.test_mode,
            'skip_sections': run_settings.skip_sections,
            'run_only': run_settings.run_only,
        }

    @staticmethod
    def _dict_to_run_settings(config_dict: dict, default_config: dict) -> RunSettings:
        # Merge default config with specific config
        merged_config = default_config.copy()
        merged_config.update(config_dict)

        run_settings = RunSettings(
            name=merged_config['name'],
            random_state=merged_config['random_state'],
            model_settings=ModelSettings(**merged_config['model_settings']),
            embedding_settings=EmbeddingSettings(**merged_config['embedding_settings']),
            data_settings=DataSettings(**merged_config['data_settings']),
            prompt_settings=PromptSettings(**merged_config['prompt_settings']),
            plot_settings=PlotSettings(**merged_config['plot_settings']),
            clustering_settings=ClusteringSettings(**merged_config['clustering_settings']),
            tsne_settings=TsneSettings(**merged_config['tsne_settings']),
            test_mode=merged_config.get('test_mode', False),
            skip_sections=merged_config.get('skip_sections', []),
            run_only=merged_config.get('run_only', None),
        )
        run_settings.update_n_clusters()
        return run_settings