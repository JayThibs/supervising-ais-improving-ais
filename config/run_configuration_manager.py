from run_settings import (
    RunSettings,
    ModelSettings,
    DataSettings,
    PlotSettings,
    PromptSettings,
    ClusteringSettings,
    TsneSettings,
)


class RunConfigurationManager:
    def __init__(self):
        """
        Initialize the run configuration manager.

        The configurations dictionary contains different run settings for the pipeline.

        reuse_data options (DataSettings): 'all', 'tsne', 'approvals', 'hierarchical', 'awareness', 'rows', 'conditions'
        hide_plots options (PlotSettings): 'all', 'tsne', 'approval', 'awareness', 'hierarchical'
        skip_sections options (RunSettings): 'model_comparison', 'approvals'
        """
        self.default_run = "quick_full_test"
        self.configurations = {
            ############################################
            # Test configurations
            ############################################
            "quick_full_test": RunSettings(
                name="quick_full_test",
                model_settings=ModelSettings(models=[("openai", "gpt-3.5-turbo")]),
                data_settings=DataSettings(n_statements=300, reuse_data=["all"]),
                test_mode=True,
            ),
            ############################################
            # Testing configurations
            # Use this to play around with different settings before saving them as a new configuration
            ############################################
            "test": RunSettings(
                name="test",
                model_settings=ModelSettings(
                    models=[
                        ("openai", "gpt-3.5-turbo"),
                        ("openai", "gpt-4"),
                        ("anthropic", "claude-v1"),
                    ]
                ),
                data_settings=DataSettings(n_statements=300, reuse_data=["all"]),
                prompt_settings=PromptSettings(
                    approval_prompts=[
                        "Briefly list the common themes of the following texts:",
                    ]
                ),
                plot_settings=PlotSettings(
                    hide_plots=["tsne", "approvals", "hierarchical", "awareness"]
                ),
                clustering_settings=ClusteringSettings(
                    n_clusters=3,
                ),
                tsne_settings=TsneSettings(perplexity=30),
            ),
            ############################################
            # Full run configurations
            ############################################
            "full_run": RunSettings(
                name="full_run",
                model_settings=ModelSettings(
                    models=[
                        ("openai", "gpt-3.5-turbo"),
                        ("openai", "gpt-4"),
                        ("anthropic", "claude-v1"),
                    ]
                ),
                data_settings=DataSettings(reuse_data=["approvals", "hierarchical"]),
                plot_settings=PlotSettings(hide_plots=["tsne"]),
            ),
            "only_model_comparisons": RunSettings(
                name="only_model_comparisons",
                model_settings=ModelSettings(
                    models=[("openai", "gpt-3.5-turbo"), ("openai", "gpt-4")]
                ),
                data_settings=DataSettings(n_statements=100, reuse_data=["all"]),
                test_mode=True,
                skip_sections=["approvals"],
            ),
        }

    def get_configuration(self, name):
        return self.configurations.get(name)
