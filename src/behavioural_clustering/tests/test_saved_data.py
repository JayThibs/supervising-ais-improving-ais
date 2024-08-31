import os
import sys
import yaml
import pickle
import unittest
from pathlib import Path
import numpy as np

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from behavioural_clustering.config.run_configuration_manager import RunConfigurationManager
from behavioural_clustering.utils.data_preparation import DataHandler

class TestSavedData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata_file = Path(project_root) / "data" / "results" / "metadata_for_runs.yaml"
        cls.run_config_manager = RunConfigurationManager()
        cls.data_handler = DataHandler(
            Path(project_root) / "data" / "results" / "pickle_files",
            Path(project_root) / "data" / "results" / "data_file_mapping.yaml"
        )

    def load_metadata(self):
        with open(self.metadata_file, 'r') as f:
            return yaml.safe_load(f)

    def test_metadata_exists(self):
        self.assertTrue(os.path.exists(self.metadata_file), "Metadata file does not exist")

    def test_run_configurations(self):
        available_runs = self.run_config_manager.list_configurations()
        self.assertGreater(len(available_runs), 0, "No run configurations found")

    def test_saved_data(self):
        metadata = self.load_metadata()
        print(f"Metadata type: {type(metadata)}")
        print(f"Number of runs: {len(metadata)}")

        for run_id, run_metadata in metadata.items():
            print(f"\nExamining run: {run_id}")
            self.assertIn('data_files', run_metadata, f"Run {run_id} is missing 'data_files' in metadata")
            self.assertIn('data_settings', run_metadata, f"Run {run_id} is missing 'data_settings' in metadata")

            data_files = run_metadata['data_files']
            for data_type, file_path in data_files.items():
                print(f"\n  Data type: {data_type}")
                print(f"  File path: {file_path}")
                
                if not os.path.exists(file_path):
                    print(f"  WARNING: File not found for {data_type}: {file_path}")
                    continue

                data = self.data_handler.load_saved_data(file_path=file_path)
                if data is None:
                    print(f"  WARNING: Failed to load data for {data_type}")
                    continue

                print(f"  Data type: {type(data)}")
                
                if isinstance(data, dict):
                    print(f"  Number of keys: {len(data)}")
                    print(f"  Keys: {list(data.keys())[:5]}...")
                elif isinstance(data, list):
                    print(f"  Number of items: {len(data)}")
                    print(f"  First item type: {type(data[0])}")
                elif isinstance(data, np.ndarray):
                    print(f"  Shape: {data.shape}")
                    print(f"  Data type: {data.dtype}")
                else:
                    print(f"  Unable to determine structure for type: {type(data)}")

                # Print a sample of the data
                print("  Data sample:")
                if isinstance(data, dict):
                    for key, value in list(data.items())[:2]:
                        print(f"    {key}: {str(value)[:100]}...")
                elif isinstance(data, list):
                    for item in data[:2]:
                        print(f"    {str(item)[:100]}...")
                elif isinstance(data, np.ndarray):
                    print(f"    {str(data.flatten()[:5])}...")
                else:
                    print(f"    {str(data)[:100]}...")

            # Check data settings
            data_settings = run_metadata['data_settings']
            print("\n  Data settings:")
            print(f"    datasets: {data_settings.get('datasets', 'Not found')}")
            print(f"    n_statements: {data_settings.get('n_statements', 'Not found')}")

if __name__ == '__main__':
    unittest.main()