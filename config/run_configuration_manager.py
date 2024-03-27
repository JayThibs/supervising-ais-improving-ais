from run_settings import RunSettings, ModelSettings, DataSettings, PlotSettings


class RunConfigurationManager:
    def __init__(self):
        self.default_run = "quick_full_test"
        self.configurations = {
            ############################################
            # Test configurations
            ############################################
            "quick_full_test": RunSettings(
                name="quick_full_test",
                model_settings=ModelSettings(models=[("openai", "gpt-3.5-turbo")]),
                data_settings=DataSettings(
                    n_statements=300, texts_subset=5, reuse_data=["all"]
                ),
                test_mode=True,
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
                data_settings=DataSettings(
                    n_statements=100, texts_subset=5, reuse_data=["all"]
                ),
                test_mode=True,
                skip_sections=["approvals"],
            ),
        }

    def get_configuration(self, name):
        return self.configurations.get(name)
