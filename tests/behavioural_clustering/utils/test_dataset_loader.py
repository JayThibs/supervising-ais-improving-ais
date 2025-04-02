import unittest
from unittest.mock import patch, mock_open, MagicMock
import os
import json
from pathlib import Path
import yaml
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.behavioural_clustering.utils.dataset_loader import (
    DatasetConfig, DatasetRegistry, DatasetLoader, create_default_registry
)


class TestDatasetConfig(unittest.TestCase):
    """Tests for the DatasetConfig class."""
    
    def test_init(self):
        """Test initialization of DatasetConfig."""
        config = DatasetConfig(name="test", path="/path/to/dataset")
        self.assertEqual(config.name, "test")
        self.assertEqual(config.path, Path("/path/to/dataset"))
        self.assertEqual(config.file_pattern, "**/*.jsonl")
        self.assertEqual(config.statement_field, "statement")
        
    def test_post_init_converts_string_path(self):
        """Test that __post_init__ converts string paths to Path objects."""
        config = DatasetConfig(name="test", path="/path/to/dataset")
        self.assertIsInstance(config.path, Path)
        
        path_obj = Path("/path/to/dataset")
        config = DatasetConfig(name="test", path=path_obj)
        self.assertIs(config.path, path_obj)


class TestDatasetRegistry(unittest.TestCase):
    """Tests for the DatasetRegistry class."""
    
    def test_init_without_config(self):
        """Test initialization without a config file."""
        registry = DatasetRegistry()
        self.assertEqual(registry.datasets, {})
        self.assertIsNone(registry.config_path)
        
    @patch('builtins.open', new_callable=mock_open, read_data=yaml.dump({
        'test_dataset': {
            'path': '/path/to/dataset',
            'file_pattern': '**/*.jsonl',
            'statement_field': 'statement'
        }
    }))
    @patch('pathlib.Path.exists', return_value=True)
    def test_init_with_config(self, mock_exists, mock_file):
        """Test initialization with a config file."""
        registry = DatasetRegistry(config_path="/path/to/config.yaml")
        self.assertEqual(len(registry.datasets), 1)
        self.assertIn('test_dataset', registry.datasets)
        self.assertEqual(registry.datasets['test_dataset'].name, 'test_dataset')
        self.assertEqual(registry.datasets['test_dataset'].path, Path('/path/to/dataset'))
        
    def test_register_dataset(self):
        """Test registering a dataset."""
        registry = DatasetRegistry()
        config = DatasetConfig(name="test", path="/path/to/dataset")
        
        registry.save_config = MagicMock()
        
        registry.register_dataset(config)
        self.assertIn('test', registry.datasets)
        self.assertEqual(registry.datasets['test'], config)
        registry.save_config.assert_called_once()
        
    def test_get_dataset(self):
        """Test getting a dataset."""
        registry = DatasetRegistry()
        config = DatasetConfig(name="test", path="/path/to/dataset")
        registry.datasets['test'] = config
        
        self.assertEqual(registry.get_dataset('test'), config)
        
        self.assertIsNone(registry.get_dataset('nonexistent'))
        
    def test_list_datasets(self):
        """Test listing datasets."""
        registry = DatasetRegistry()
        registry.datasets = {
            'test1': DatasetConfig(name="test1", path="/path/to/dataset1"),
            'test2': DatasetConfig(name="test2", path="/path/to/dataset2")
        }
        
        datasets = registry.list_datasets()
        self.assertEqual(len(datasets), 2)
        self.assertIn('test1', datasets)
        self.assertIn('test2', datasets)


class TestDatasetLoader(unittest.TestCase):
    """Tests for the DatasetLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = DatasetRegistry()
        self.loader = DatasetLoader(self.registry)
        
    def test_init(self):
        """Test initialization."""
        self.assertEqual(self.loader.registry, self.registry)
        
    def test_load_dataset_not_found(self):
        """Test loading a dataset that doesn't exist."""
        result, accepted, rejected = self.loader.load_dataset("nonexistent")
        self.assertEqual(result, [])
        self.assertEqual(accepted, 0)
        self.assertEqual(rejected, 0)
        
    @patch('glob.glob', return_value=['/path/to/dataset/file1.jsonl', '/path/to/dataset/file2.jsonl'])
    def test_load_dataset(self, mock_glob):
        """Test loading a dataset."""
        self.loader._load_file = MagicMock(side_effect=[
            ([{'statement': 'test1', 'metadata': {}}], True),
            ([{'statement': 'test2', 'metadata': {}}], True)
        ])
        
        config = DatasetConfig(name="test", path="/path/to/dataset")
        self.registry.datasets['test'] = config
        
        result, accepted, rejected = self.loader.load_dataset("test")
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['statement'], 'test1')
        self.assertEqual(result[1]['statement'], 'test2')
        self.assertEqual(accepted, 2)
        self.assertEqual(rejected, 0)
        
    @patch('builtins.open', new_callable=mock_open, read_data='{"statement": "test1"}\n{"statement": "test2"}')
    @patch('pathlib.Path.exists', return_value=True)
    def test_load_file(self, mock_exists, mock_file):
        """Test loading a file."""
        config = DatasetConfig(name="test", path="/path/to/dataset")
        result, accepted = self.loader._load_file(Path("/path/to/dataset/file.jsonl"), config)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['statement'], 'test1')
        self.assertEqual(result[1]['statement'], 'test2')
        self.assertTrue(accepted)
        
    def test_filter_statements(self):
        """Test filtering statements."""
        statements = [
            {'statement': 'short', 'metadata': {'category': 'A'}},
            {'statement': 'medium length statement', 'metadata': {'category': 'B'}},
            {'statement': 'very long statement with lots of words', 'metadata': {'category': 'A'}}
        ]
        
        filtered = self.loader.filter_statements(statements, max_length=10)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]['statement'], 'short')
        
        filtered = self.loader.filter_statements(statements, min_length=15)
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0]['statement'], 'medium length statement')
        self.assertEqual(filtered[1]['statement'], 'very long statement with lots of words')
        
        filtered = self.loader.filter_statements(statements, categories=['A'])
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0]['statement'], 'short')
        self.assertEqual(filtered[1]['statement'], 'very long statement with lots of words')
        
        filtered = self.loader.filter_statements(statements, min_length=15, categories=['A'])
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]['statement'], 'very long statement with lots of words')


class TestCreateDefaultRegistry(unittest.TestCase):
    """Tests for the create_default_registry function."""
    
    @patch('src.behavioural_clustering.utils.dataset_loader.DatasetRegistry')
    def test_create_default_registry(self, mock_registry_class):
        """Test creating a default registry."""
        mock_registry = MagicMock()
        mock_registry.datasets = {}
        mock_registry_class.return_value = mock_registry
        
        result = create_default_registry("/path/to/data")
        
        mock_registry_class.assert_called_once_with(Path("/path/to/data") / "dataset_config.yaml")
        
        self.assertEqual(mock_registry.register_dataset.call_count, 3)
        
        self.assertEqual(result, mock_registry)
        
        mock_registry.datasets = {'existing': 'dataset'}
        result = create_default_registry("/path/to/data")
        
        self.assertEqual(mock_registry.register_dataset.call_count, 3)  # Still 3 from before


if __name__ == '__main__':
    unittest.main()
