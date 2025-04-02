# Testing the Behavioural Clustering Pipeline

This document describes how to run and write tests for the behavioural clustering pipeline.

## Quick Start

Run all tests:
```bash
pytest tests/
```

Run only fast tests:
```bash
pytest tests/ --runslow=false
```

Run integration tests:
```bash
pytest tests/integration/
```

Run specific test file:
```bash
pytest tests/integration/test_evaluation_pipeline.py
```

## Test Structure

```
tests/
├── integration/
│   └── test_evaluation_pipeline.py  # Full pipeline tests
├── unit/
│   ├── test_clustering.py
│   ├── test_visualization.py
│   └── ...
└── conftest.py                      # Shared test configuration
```

## Test Categories

### Integration Tests
- `test_standard_evaluation_pipeline_basic`: Verifies basic standard evaluation
- `test_iterative_evaluation_pipeline_basic`: Verifies basic iterative evaluation
- `test_standard_evaluation_output_files`: Checks output file generation
- `test_iterative_evaluation_output_files`: Checks iterative output files
- `test_error_handling`: Verifies error cases

### Parameterized Tests
- Tests with different prompt counts
- Tests with different iteration counts
- Tests with various configurations

## Writing Tests

### Standard Pipeline Test Example
```python
def test_custom_standard_evaluation(mock_run_settings):
    pipeline = EvaluatorPipeline(mock_run_settings)
    pipeline.run_evaluations()
    
    # Verify results
    assert pipeline.run_id is not None
    assert Path(mock_run_settings.directory_settings.results_dir).exists()
```

### Iterative Pipeline Test Example
```python
def test_custom_iterative_evaluation(mock_iterative_settings):
    pipeline = EvaluatorPipeline(mock_iterative_settings)
    pipeline.run_iterative_evaluation()
    
    # Verify results
    iterative_dir = Path(mock_iterative_settings.directory_settings.data_dir) / "iterative"
    assert iterative_dir.exists()
```

## Fixtures

### `mock_run_settings`
- Creates minimal run settings for testing
- Sets up temporary directories
- Configures basic model and data settings

### `mock_iterative_settings`
- Extends `mock_run_settings`
- Adds iterative analysis configuration
- Sets up iterative-specific paths

## Test Data

Test data is managed through temporary directories:
- Created fresh for each test class
- Automatically cleaned up after tests
- Structured to match production layout

## Adding New Tests

1. Create test file in appropriate directory
2. Import necessary fixtures
3. Write test functions
4. Add parameterization if needed
5. Verify cleanup

Example:
```python
@pytest.mark.parametrize("model_count", [1, 2, 3])
def test_multiple_models(mock_run_settings, model_count):
    settings_dict = mock_run_settings.to_dict()
    settings_dict["model_settings"]["models"] = [
        ["anthropic", f"model-{i}"] for i in range(model_count)
    ]
    settings = RunSettings.from_dict(settings_dict)
    
    pipeline = EvaluatorPipeline(settings)
    pipeline.run_evaluations()
    
    # Verify results for each model
    ...
```

## Best Practices

1. Use fixtures for common setup
2. Clean up test data
3. Test both success and failure cases
4. Verify output files and data
5. Use parameterization for variations
6. Add appropriate markers
7. Include meaningful assertions
8. Document test purpose

## Troubleshooting

### Common Issues

1. **Missing Test Data**
   - Verify fixtures are working
   - Check temporary directory creation

2. **Slow Tests**
   - Use `--runslow` flag appropriately
   - Consider reducing test data size

3. **Failed Cleanup**
   - Manual cleanup: `rm -rf tests/tmp*`
   - Check fixture teardown

4. **Resource Issues**
   - Reduce batch sizes in test settings
   - Use smaller model configurations

### Debug Tips

1. Use pytest's `-v` flag for verbose output:
```bash
pytest -v tests/integration/
```

2. Debug specific test:
```bash
pytest -v tests/integration/test_evaluation_pipeline.py::TestEvaluationPipeline::test_standard_evaluation_pipeline_basic
```

3. Show print output:
```bash
pytest -s tests/integration/
```