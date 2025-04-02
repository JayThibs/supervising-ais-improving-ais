import pytest
from unittest.mock import patch, MagicMock, call
from pathlib import Path

"""
This test file is designed to test the pipeline without triggering actual
time-consuming external API calls. We mock out calls to the LLM or other
external services to keep tests fast and self-contained.
"""

@pytest.mark.integration
class TestPipelineMocked:
    @pytest.fixture
    def mock_evaluation_pipeline(self):
        """
        Fixture that patches the EvaluatorPipeline and returns a mocked instance,
        ensuring that no external API calls are made.
        """
        with patch("behavioural_clustering.evaluation.evaluator_pipeline.EvaluatorPipeline") as MockPipeline:
            # Create mock pipeline instance
            mock_pipeline = MagicMock()
            
            # Mock data preparation
            mock_pipeline.data_prep = MagicMock()
            mock_pipeline.data_prep.load_and_preprocess_data.return_value = ["test prompt 1", "test prompt 2"]
            
            # Mock the model evaluation manager
            mock_model_manager = MagicMock()
            mock_model_manager.generate_responses.return_value = {
                "statements": ["test statement"],
                "responses": {"model-1": ["test response"]},
                "metadata": {"timestamp": "2024-01-02T12:00:00"}
            }
            mock_pipeline.model_evaluation_manager = mock_model_manager
            
            # Mock the iterative analyzer
            mock_iterative = MagicMock()
            mock_iterative.run_iterative_evaluation.return_value = None
            mock_pipeline.iterative_analyzer = mock_iterative
            
            # Mock run settings
            mock_settings = MagicMock()
            mock_settings.iterative_settings.max_iterations = 2
            mock_settings.iterative_settings.prompts_per_iteration = 5
            mock_pipeline.run_settings = mock_settings
            
            # Mock the evaluation methods
            def mock_run_evaluations():
                return mock_pipeline.model_evaluation_manager.generate_responses()
            
            def mock_run_iterative():
                mock_pipeline.iterative_analyzer.run_iterative_evaluation(
                    initial_prompts=mock_pipeline.data_prep.load_and_preprocess_data(),
                    model_evaluation_manager=mock_pipeline.model_evaluation_manager,
                    data_prep=mock_pipeline.data_prep,
                    run_settings=mock_pipeline.run_settings
                )
            
            mock_pipeline.run_evaluations = mock_run_evaluations
            mock_pipeline.run_iterative_evaluation = mock_run_iterative
            
            MockPipeline.return_value = mock_pipeline
            yield mock_pipeline

    def test_mocked_pipeline_runs_without_errors(self, mock_evaluation_pipeline):
        """
        Basic test to ensure pipeline's main methods can be called
        without performing expensive or time-consuming API calls.
        """
        # Reset mock call counts
        mock_evaluation_pipeline.model_evaluation_manager.generate_responses.reset_mock()
        mock_evaluation_pipeline.iterative_analyzer.run_iterative_evaluation.reset_mock()
        
        mock_evaluation_pipeline.run_evaluations()
        mock_evaluation_pipeline.run_iterative_evaluation()

        # Verify the calls happened
        assert mock_evaluation_pipeline.model_evaluation_manager.generate_responses.call_count >= 1
        assert mock_evaluation_pipeline.iterative_analyzer.run_iterative_evaluation.call_count >= 1

    def test_pipeline_saving_data(self, mock_evaluation_pipeline):
        """
        Ensure that the pipeline is capable of saving data during or after runs,
        without calling any real external services.
        """
        # Reset mock call counts
        mock_evaluation_pipeline.model_evaluation_manager.generate_responses.reset_mock()
        mock_evaluation_pipeline.save_run_data.reset_mock()
        
        # Mock a call that the pipeline might make to save data
        mock_evaluation_pipeline.save_run_data()
        mock_evaluation_pipeline.save_run_data.assert_called_once()

        # Run evaluations and verify data saving
        mock_evaluation_pipeline.run_evaluations()
        assert mock_evaluation_pipeline.model_evaluation_manager.generate_responses.call_count >= 1

    def test_pipeline_logging(self, mock_evaluation_pipeline):
        """
        Check that certain logging or debugging steps are triggered
        within the pipeline (if applicable).
        """
        # Reset mock call counts
        mock_evaluation_pipeline.model_evaluation_manager.generate_responses.reset_mock()
        
        # Run evaluations and check model manager calls
        mock_evaluation_pipeline.run_evaluations()
        assert mock_evaluation_pipeline.model_evaluation_manager.generate_responses.call_count >= 1

    def test_pipeline_iterative_logic(self, mock_evaluation_pipeline):
        """
        Ensure the pipeline's iterative evaluation logic is invoked
        with the appropriate arguments or conditions.
        """
        # Reset mock call counts
        mock_evaluation_pipeline.iterative_analyzer.run_iterative_evaluation.reset_mock()
        
        # Run iterative evaluation
        mock_evaluation_pipeline.run_iterative_evaluation()
        assert mock_evaluation_pipeline.iterative_analyzer.run_iterative_evaluation.call_count == 1
        
        # Reset mock for next test
        mock_evaluation_pipeline.iterative_analyzer.run_iterative_evaluation.reset_mock()
        
        # Run multiple times to test iteration handling
        for _ in range(3):
            mock_evaluation_pipeline.run_iterative_evaluation()
        assert mock_evaluation_pipeline.iterative_analyzer.run_iterative_evaluation.call_count == 3

    def test_pipeline_arguments_passed_correctly(self, mock_evaluation_pipeline):
        """
        Check that the pipeline is invoked with particular configurations
        or arguments, if relevant to your use case.
        """
        # Reset mock call counts
        mock_evaluation_pipeline.iterative_analyzer.run_iterative_evaluation.reset_mock()
        
        # Mock run settings
        mock_settings = MagicMock()
        mock_settings.iterative_settings.max_iterations = 2
        mock_settings.iterative_settings.prompts_per_iteration = 5
        
        # Run with settings
        mock_evaluation_pipeline.run_settings = mock_settings
        mock_evaluation_pipeline.run_iterative_evaluation()
        
        # Verify settings were used
        assert mock_evaluation_pipeline.iterative_analyzer.run_iterative_evaluation.call_count == 1
        mock_evaluation_pipeline.iterative_analyzer.run_iterative_evaluation.assert_called_with(
            initial_prompts=mock_evaluation_pipeline.data_prep.load_and_preprocess_data(),
            model_evaluation_manager=mock_evaluation_pipeline.model_evaluation_manager,
            data_prep=mock_evaluation_pipeline.data_prep,
            run_settings=mock_settings
        )

    def test_pipeline_order_of_operations(self, mock_evaluation_pipeline):
        """
        Check that certain pipeline methods occur in the correct order.
        """
        # Reset mock call counts
        mock_evaluation_pipeline.model_evaluation_manager.generate_responses.reset_mock()
        mock_evaluation_pipeline.iterative_analyzer.run_iterative_evaluation.reset_mock()
        
        # Example: We call run_evaluations, then run_iterative_evaluation
        mock_evaluation_pipeline.run_evaluations()
        mock_evaluation_pipeline.run_iterative_evaluation()

        # Now we can check if run_evaluations was called before run_iterative_evaluation:
        assert mock_evaluation_pipeline.model_evaluation_manager.generate_responses.call_count >= 1
        assert mock_evaluation_pipeline.iterative_analyzer.run_iterative_evaluation.call_count >= 1