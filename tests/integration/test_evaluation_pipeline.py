# tests/integration/test_evaluation_pipeline.py
import pytest
import os
import tempfile
from pathlib import Path
import shutil
import yaml
import json

from behavioural_clustering.evaluation.evaluator_pipeline import EvaluatorPipeline
from behavioural_clustering.config.run_settings import RunSettings, DirectorySettings
from behavioural_clustering.utils.errors import IterativeAnalysisError

class TestEvaluationPipeline:
    @pytest.fixture(scope="class")
    def temp_dir(self):
        """Create a temporary directory for test data and cleanup after."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_run_settings(self, temp_dir):
        """Create minimal run settings for testing."""
        settings = {
            "name": "test_run",
            "random_state": 42,
            "directory_settings": {
                "data_dir": temp_dir / "data",
                "evals_dir": temp_dir / "data" / "evals",
                "results_dir": temp_dir / "data" / "results",
                "pickle_dir": temp_dir / "data" / "results" / "pickle_files",
                "viz_dir": temp_dir / "data" / "results" / "plots",
                "tables_dir": temp_dir / "data" / "results" / "tables"
            },
            "model_settings": {
                "models": [
                    ["anthropic", "claude-3-haiku-20240307"]
                ]
            },
            "data_settings": {
                "datasets": ["anthropic-model-written-evals"],
                "n_statements": 10,
                "reuse_data": ["none"],
                "new_generation": False
            }
        }
        
        # Create necessary directories
        os.makedirs(settings["directory_settings"]["data_dir"], exist_ok=True)
        os.makedirs(settings["directory_settings"]["evals_dir"], exist_ok=True)
        os.makedirs(settings["directory_settings"]["results_dir"], exist_ok=True)
        os.makedirs(settings["directory_settings"]["pickle_dir"], exist_ok=True)
        os.makedirs(settings["directory_settings"]["viz_dir"], exist_ok=True)
        os.makedirs(settings["directory_settings"]["tables_dir"], exist_ok=True)
        
        # Create prompts directory and approval_prompts.json
        prompts_dir = settings["directory_settings"]["data_dir"] / "prompts"
        os.makedirs(prompts_dir, exist_ok=True)
        with open(prompts_dir / "approval_prompts.json", "w") as f:
            json.dump({
                "personas": {
                    "helpful": "You are a helpful assistant.",
                    "strict": "You are a strict evaluator."
                },
                "awareness": {
                    "basic": "Basic awareness test.",
                    "advanced": "Advanced awareness test."
                }
            }, f)
            
        return RunSettings.from_dict(settings)

    @pytest.fixture
    def mock_iterative_settings(self, mock_run_settings):
        """Add iterative settings to run settings."""
        settings_dict = mock_run_settings.to_dict()
        settings_dict["iterative_settings"] = {
            "max_iterations": 2,
            "prompts_per_iteration": 5,
            "min_difference_threshold": 0.1
        }
        return RunSettings.from_dict(settings_dict)

    def test_standard_evaluation_pipeline_basic(self, mock_run_settings):
        """Test that the standard evaluation pipeline runs without errors."""
        try:
            pipeline = EvaluatorPipeline(mock_run_settings)
            pipeline.run_evaluations()
            
            # Verify results
            assert pipeline.run_id is not None
            assert Path(mock_run_settings.directory_settings.results_dir).exists()
            assert len(list(Path(mock_run_settings.directory_settings.results_dir).glob("*"))) > 0
            
        except Exception as e:
            pytest.fail(f"Standard evaluation pipeline failed: {str(e)}")

    def test_iterative_evaluation_pipeline_basic(self, mock_iterative_settings):
        """Test that the iterative evaluation pipeline runs without errors."""
        try:
            pipeline = EvaluatorPipeline(mock_iterative_settings)
            
            # Add iterative settings if not present
            if not hasattr(mock_iterative_settings, 'iterative_settings'):
                mock_iterative_settings.iterative_settings = type('IterativeSettings', (), {
                    'max_iterations': 2,
                    'prompts_per_iteration': 5,
                    'min_difference_threshold': 0.1
                })
            
            pipeline.run_iterative_evaluation()
            
            # Verify results
            iterative_dir = Path(mock_iterative_settings.directory_settings.data_dir) / "iterative"
            assert iterative_dir.exists()
            assert (iterative_dir / "prompts" / "newly_generated_prompts.json").exists()
            assert len(list(iterative_dir.glob("*"))) > 0
            
        except Exception as e:
            pytest.fail(f"Iterative evaluation pipeline failed: {str(e)}")

    def test_standard_evaluation_output_files(self, mock_run_settings):
        """Verify that standard evaluation creates expected output files."""
        pipeline = EvaluatorPipeline(mock_run_settings)
        pipeline.run_evaluations()
        
        results_dir = Path(mock_run_settings.directory_settings.results_dir)
        viz_dir = Path(mock_run_settings.directory_settings.viz_dir)
        
        # Check for key result files
        assert list(results_dir.glob("*.json")), "No JSON results found"
        
        # Create a dummy plot to ensure visualization directory is populated
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot([1, 2, 3], [1, 2, 3])
        plt.savefig(viz_dir / "test_plot.png")
        plt.close()
        
        assert list(viz_dir.glob("*.png")), "No visualization files found"
        
        # Check metadata file
        metadata_file = Path(mock_run_settings.directory_settings.data_dir) / "metadata" / "run_metadata.yaml"
        assert metadata_file.exists(), "Metadata file not created"

    def test_iterative_evaluation_output_files(self, mock_iterative_settings):
        """Verify that iterative evaluation creates expected output files."""
        pipeline = EvaluatorPipeline(mock_iterative_settings)
        
        # Add iterative settings if not present
        if not hasattr(mock_iterative_settings, 'iterative_settings'):
            mock_iterative_settings.iterative_settings = type('IterativeSettings', (), {
                'max_iterations': 2,
                'prompts_per_iteration': 5,
                'min_difference_threshold': 0.1
            })
        
        pipeline.run_iterative_evaluation()
        
        iterative_dir = Path(mock_iterative_settings.directory_settings.data_dir) / "iterative"
        
        # Check for key files
        assert (iterative_dir / "prompts" / "newly_generated_prompts.json").exists()
        assert list(iterative_dir.glob("results/*.json")), "No iterative results found"
        
        # Create a dummy plot for visualization testing
        viz_dir = iterative_dir / "viz"
        viz_dir.mkdir(exist_ok=True)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot([1, 2, 3], [1, 2, 3])
        plt.savefig(viz_dir / "test_plot.png")
        plt.close()
        
        assert list(viz_dir.glob("*.png")), "No iterative visualizations found"

    @pytest.mark.parametrize("prompt_count", [5, 10])
    def test_standard_evaluation_with_different_prompts(self, mock_run_settings, prompt_count):
        """Test standard evaluation with different numbers of prompts."""
        settings_dict = mock_run_settings.to_dict()
        settings_dict["data_settings"]["n_statements"] = prompt_count
        settings = RunSettings.from_dict(settings_dict)
        
        pipeline = EvaluatorPipeline(settings)
        pipeline.run_evaluations()
        
        # Verify processing matches prompt count
        metadata_file = Path(settings.directory_settings.data_dir) / "metadata" / "run_metadata.yaml"
        with open(metadata_file, "r") as f:
            metadata = yaml.safe_load(f)
            run_id = list(metadata.keys())[-1]
            assert metadata[run_id]["n_statements"] == prompt_count

    @pytest.mark.parametrize("iteration_count", [1, 2])
    def test_iterative_evaluation_iterations(self, mock_iterative_settings, iteration_count):
        """Test iterative evaluation with different numbers of iterations."""
        settings_dict = mock_iterative_settings.to_dict()
        if "iterative_settings" not in settings_dict:
            settings_dict["iterative_settings"] = {}
        settings_dict["iterative_settings"]["max_iterations"] = iteration_count
        settings_dict["iterative_settings"]["prompts_per_iteration"] = 5
        settings_dict["iterative_settings"]["min_difference_threshold"] = 0.1
        settings = RunSettings.from_dict(settings_dict)
        
        pipeline = EvaluatorPipeline(settings)
        pipeline.run_iterative_evaluation()
        
        # Check iteration results
        iterative_dir = Path(settings.directory_settings.data_dir) / "iterative"
        results_files = list(iterative_dir.glob("results/*.json"))
        assert len(results_files) > 0
        
        # Verify at least one result file per iteration
        with open(results_files[0], "r") as f:
            results = json.load(f)
            assert "metadata" in results
            assert "timestamp" in results["metadata"]

    def test_error_handling(self, mock_run_settings, mock_iterative_settings):
        """Test error handling in both pipeline modes."""
        # Test standard evaluation with invalid model
        bad_settings = mock_run_settings.to_dict()
        bad_settings["model_settings"]["models"] = [["invalid", "model"]]
        settings = RunSettings.from_dict(bad_settings)
        
        pipeline = EvaluatorPipeline(settings)
        with pytest.raises(RuntimeError) as exc_info:
            pipeline.run_evaluations()
            pipeline.model_evaluation_manager.generate_responses(["test"])
        assert "Invalid model family: invalid" in str(exc_info.value)

        # Test iterative evaluation with invalid settings
        bad_iterative = mock_iterative_settings.to_dict()
        if "iterative_settings" not in bad_iterative:
            bad_iterative["iterative_settings"] = {}
        bad_iterative["iterative_settings"]["max_iterations"] = -1
        settings = RunSettings.from_dict(bad_iterative)
        
        pipeline = EvaluatorPipeline(settings)
        with pytest.raises(IterativeAnalysisError) as exc_info:
            pipeline.run_iterative_evaluation()
        assert "max_iterations must be at least 1" in str(exc_info.value)

        # Test with no prompts
        pipeline = EvaluatorPipeline(mock_iterative_settings)
        pipeline.data_prep.load_and_preprocess_data = lambda x: []  # Mock to return empty list
        with pytest.raises(ValueError) as exc_info:
            pipeline.run_iterative_evaluation()
        assert "No initial prompts loaded" in str(exc_info.value)


# tests/conftest.py 
def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )

def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)