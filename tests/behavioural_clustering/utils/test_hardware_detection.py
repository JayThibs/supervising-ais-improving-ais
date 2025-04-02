import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.behavioural_clustering.utils.hardware_detection import (
    HardwareInfo, get_hardware_info, configure_models_for_hardware
)


class TestHardwareInfo(unittest.TestCase):
    """Tests for the HardwareInfo class."""
    
    @patch('src.behavioural_clustering.utils.hardware_detection.platform')
    @patch('src.behavioural_clustering.utils.hardware_detection.psutil')
    @patch('src.behavioural_clustering.utils.hardware_detection.torch')
    def test_init(self, mock_torch, mock_psutil, mock_platform):
        """Test initialization."""
        mock_platform.platform.return_value = "Linux-5.4.0-x86_64"
        mock_platform.system.return_value = "Linux"
        mock_platform.release.return_value = "5.4.0"
        mock_platform.version.return_value = "#1 SMP"
        mock_platform.machine.return_value = "x86_64"
        mock_platform.processor.return_value = "x86_64"
        
        mock_psutil.cpu_count.side_effect = [4, 8]  # physical, logical
        mock_psutil.cpu_percent.return_value = 10.0
        mock_cpu_freq = MagicMock()
        mock_cpu_freq._asdict.return_value = {'current': 2.5, 'min': 1.0, 'max': 3.0}
        mock_psutil.cpu_freq.return_value = mock_cpu_freq
        
        mock_memory = MagicMock()
        mock_memory.total = 16 * 1024 ** 3
        mock_memory.available = 8 * 1024 ** 3
        mock_memory.used = 8 * 1024 ** 3
        mock_memory.percent = 50.0
        mock_psutil.virtual_memory.return_value = mock_memory
        
        mock_torch.cuda.is_available.return_value = True
        mock_torch.version.cuda = "11.1"
        mock_torch.cuda.device_count.return_value = 2
        
        mock_device1 = MagicMock()
        mock_device1.name = "GeForce RTX 3080"
        mock_device1.total_memory = 10 * 1024 ** 3
        mock_device1.major = 8
        mock_device1.minor = 6
        
        mock_device2 = MagicMock()
        mock_device2.name = "GeForce RTX 3070"
        mock_device2.total_memory = 8 * 1024 ** 3
        mock_device2.major = 8
        mock_device2.minor = 6
        
        mock_torch.cuda.get_device_properties.side_effect = [mock_device1, mock_device2]
        
        hardware = HardwareInfo()
        
        self.assertEqual(hardware.system_info['platform'], "Linux-5.4.0-x86_64")
        self.assertEqual(hardware.system_info['system'], "Linux")
        self.assertEqual(hardware.system_info['release'], "5.4.0")
        self.assertEqual(hardware.system_info['version'], "#1 SMP")
        self.assertEqual(hardware.system_info['machine'], "x86_64")
        self.assertEqual(hardware.system_info['processor'], "x86_64")
        
        self.assertEqual(hardware.cpu_info['physical_cores'], 4)
        self.assertEqual(hardware.cpu_info['logical_cores'], 8)
        self.assertEqual(hardware.cpu_info['cpu_percent'], 10.0)
        self.assertEqual(hardware.cpu_info['cpu_freq'], {'current': 2.5, 'min': 1.0, 'max': 3.0})
        
        self.assertEqual(hardware.memory_info['total'], 16 * 1024 ** 3)
        self.assertEqual(hardware.memory_info['available'], 8 * 1024 ** 3)
        self.assertEqual(hardware.memory_info['used'], 8 * 1024 ** 3)
        self.assertEqual(hardware.memory_info['percent'], 50.0)
        self.assertEqual(hardware.memory_info['total_gb'], 16.0)
        self.assertEqual(hardware.memory_info['available_gb'], 8.0)
        
        self.assertEqual(len(hardware.gpu_info), 2)
        self.assertEqual(hardware.gpu_info[0]['index'], 0)
        self.assertEqual(hardware.gpu_info[0]['name'], "GeForce RTX 3080")
        self.assertEqual(hardware.gpu_info[0]['total_memory'], 10 * 1024 ** 3)
        self.assertEqual(hardware.gpu_info[0]['total_memory_gb'], 10.0)
        self.assertEqual(hardware.gpu_info[0]['cuda_version'], "11.1")
        self.assertEqual(hardware.gpu_info[0]['compute_capability'], "8.6")
        
        self.assertEqual(hardware.gpu_info[1]['index'], 1)
        self.assertEqual(hardware.gpu_info[1]['name'], "GeForce RTX 3070")
        self.assertEqual(hardware.gpu_info[1]['total_memory'], 8 * 1024 ** 3)
        self.assertEqual(hardware.gpu_info[1]['total_memory_gb'], 8.0)
        self.assertEqual(hardware.gpu_info[1]['cuda_version'], "11.1")
        self.assertEqual(hardware.gpu_info[1]['compute_capability'], "8.6")
    
    @patch('src.behavioural_clustering.utils.hardware_detection.torch')
    def test_get_optimal_device_with_cuda(self, mock_torch):
        """Test get_optimal_device with CUDA available."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        
        mock_device1 = MagicMock()
        mock_device1.total_memory = 10 * 1024 ** 3
        
        mock_device2 = MagicMock()
        mock_device2.total_memory = 8 * 1024 ** 3
        
        mock_torch.cuda.get_device_properties.side_effect = [mock_device1, mock_device2]
        mock_torch.cuda.memory_allocated.side_effect = [2 * 1024 ** 3, 1 * 1024 ** 3]
        
        hardware = HardwareInfo()
        hardware.gpu_info = [
            {'index': 0, 'name': 'GeForce RTX 3080', 'total_memory': 10 * 1024 ** 3, 'total_memory_gb': 10.0},
            {'index': 1, 'name': 'GeForce RTX 3070', 'total_memory': 8 * 1024 ** 3, 'total_memory_gb': 8.0}
        ]
        
        device = hardware.get_optimal_device()
        
        self.assertEqual(device, "cuda:0")
        
    @patch('src.behavioural_clustering.utils.hardware_detection.torch')
    def test_get_optimal_device_without_cuda(self, mock_torch):
        """Test get_optimal_device without CUDA available."""
        mock_torch.cuda.is_available.return_value = False
        
        hardware = HardwareInfo()
        hardware.gpu_info = []
        
        device = hardware.get_optimal_device()
        
        self.assertEqual(device, "cpu")
    
    @patch('src.behavioural_clustering.utils.hardware_detection.torch')
    def test_get_optimal_batch_size_gpu(self, mock_torch):
        """Test get_optimal_batch_size with GPU."""
        hardware = HardwareInfo()
        hardware.gpu_info = [
            {'index': 0, 'name': 'GeForce RTX 3080', 'total_memory': 10 * 1024 ** 3, 'total_memory_gb': 10.0},
            {'index': 1, 'name': 'GeForce RTX 3070', 'total_memory': 8 * 1024 ** 3, 'total_memory_gb': 8.0}
        ]
        
        batch_size = hardware.get_optimal_batch_size(model_size_gb=7.0, device="cuda:0")
        
        self.assertEqual(batch_size, 1)
        
        batch_size = hardware.get_optimal_batch_size(model_size_gb=3.0, device="cuda:0")
        
        self.assertEqual(batch_size, 8)
        
        batch_size = hardware.get_optimal_batch_size(model_size_gb=1.0, device="cuda:0")
        
        self.assertEqual(batch_size, 12)
    
    def test_get_optimal_batch_size_cpu(self):
        """Test get_optimal_batch_size with CPU."""
        hardware = HardwareInfo()
        hardware.memory_info = {
            'total_gb': 16.0,
            'available_gb': 8.0
        }
        
        batch_size = hardware.get_optimal_batch_size(model_size_gb=7.0, device="cpu")
        
        self.assertEqual(batch_size, 1)
        
        batch_size = hardware.get_optimal_batch_size(model_size_gb=3.0, device="cpu")
        
        self.assertEqual(batch_size, 1)
        
        batch_size = hardware.get_optimal_batch_size(model_size_gb=1.0, device="cpu")
        
        self.assertEqual(batch_size, 3)
    
    @patch('src.behavioural_clustering.utils.hardware_detection.torch')
    def test_get_optimal_model_config_gpu(self, mock_torch):
        """Test get_optimal_model_config with GPU."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        
        mock_device = MagicMock()
        mock_device.total_memory = 10 * 1024 ** 3
        
        mock_torch.cuda.get_device_properties.return_value = mock_device
        mock_torch.cuda.memory_allocated.return_value = 0
        
        hardware = HardwareInfo()
        hardware.gpu_info = [
            {'index': 0, 'name': 'GeForce RTX 3080', 'total_memory': 10 * 1024 ** 3, 'total_memory_gb': 10.0}
        ]
        
        available_models = [
            {'name': 'small', 'size_gb': 3.0, 'min_memory_gb': 4.0},
            {'name': 'medium', 'size_gb': 7.0, 'min_memory_gb': 8.0},
            {'name': 'large', 'size_gb': 13.0, 'min_memory_gb': 16.0}
        ]
        
        hardware.get_optimal_device = MagicMock(return_value="cuda:0")
        
        config = hardware.get_optimal_model_config(available_models)
        
        self.assertEqual(config['name'], 'medium')
        self.assertEqual(config['device'], 'cuda:0')
    
    def test_get_optimal_model_config_cpu(self):
        """Test get_optimal_model_config with CPU."""
        hardware = HardwareInfo()
        hardware.memory_info = {
            'total_gb': 16.0,
            'available_gb': 8.0
        }
        
        available_models = [
            {'name': 'small', 'size_gb': 3.0, 'min_memory_gb': 4.0},
            {'name': 'medium', 'size_gb': 7.0, 'min_memory_gb': 8.0},
            {'name': 'large', 'size_gb': 13.0, 'min_memory_gb': 16.0}
        ]
        
        hardware.get_optimal_device = MagicMock(return_value="cpu")
        
        config = hardware.get_optimal_model_config(available_models)
        
        self.assertEqual(config['name'], 'small')
        self.assertEqual(config['device'], 'cpu')
    
    @patch('src.behavioural_clustering.utils.hardware_detection.torch')
    def test_get_parallel_model_configs(self, mock_torch):
        """Test get_parallel_model_configs."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        
        hardware = HardwareInfo()
        hardware.gpu_info = [
            {'index': 0, 'name': 'GeForce RTX 3080', 'total_memory': 10 * 1024 ** 3, 'total_memory_gb': 10.0},
            {'index': 1, 'name': 'GeForce RTX 3070', 'total_memory': 8 * 1024 ** 3, 'total_memory_gb': 8.0}
        ]
        
        available_models = [
            {'name': 'small', 'size_gb': 3.0, 'min_memory_gb': 4.0},
            {'name': 'medium', 'size_gb': 7.0, 'min_memory_gb': 8.0},
            {'name': 'large', 'size_gb': 13.0, 'min_memory_gb': 16.0}
        ]
        
        hardware.get_optimal_batch_size = MagicMock(side_effect=[4, 2])
        
        configs = hardware.get_parallel_model_configs(available_models, 'medium')
        
        self.assertEqual(len(configs), 2)
        self.assertEqual(configs[0]['name'], 'medium')
        self.assertEqual(configs[0]['device'], 'cuda:0')
        self.assertEqual(configs[0]['batch_size'], 4)
        
        self.assertEqual(configs[1]['name'], 'medium')
        self.assertEqual(configs[1]['device'], 'cuda:1')
        self.assertEqual(configs[1]['batch_size'], 2)
        
        hardware.get_optimal_batch_size = MagicMock(return_value=2)
        
        hardware.gpu_info[1]['total_memory_gb'] = 5.0
        
        configs = hardware.get_parallel_model_configs(available_models, 'medium')
        
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0]['name'], 'medium')
        self.assertEqual(configs[0]['device'], 'cuda:0')
        self.assertEqual(configs[0]['batch_size'], 2)
        
        configs = hardware.get_parallel_model_configs(available_models, 'nonexistent')
        
        self.assertEqual(len(configs), 0)


class TestGetHardwareInfo(unittest.TestCase):
    """Tests for the get_hardware_info function."""
    
    @patch('src.behavioural_clustering.utils.hardware_detection.HardwareInfo')
    def test_get_hardware_info(self, mock_hardware_info_class):
        """Test get_hardware_info function."""
        mock_hardware_info = MagicMock()
        mock_hardware_info_class.return_value = mock_hardware_info
        
        result = get_hardware_info()
        
        mock_hardware_info_class.assert_called_once()
        
        self.assertEqual(result, mock_hardware_info)


class TestConfigureModelsForHardware(unittest.TestCase):
    """Tests for the configure_models_for_hardware function."""
    
    @patch('src.behavioural_clustering.utils.hardware_detection.get_hardware_info')
    @patch('src.behavioural_clustering.utils.hardware_detection.Path.exists')
    @patch('builtins.open')
    @patch('yaml.safe_load')
    def test_configure_models_for_hardware(self, mock_yaml_load, mock_open, mock_exists, mock_get_hardware_info):
        """Test configure_models_for_hardware function."""
        mock_hardware = MagicMock()
        mock_hardware.system_info = {'system': 'Linux'}
        mock_hardware.cpu_info = {'physical_cores': 4}
        mock_hardware.memory_info = {'total_gb': 16.0}
        mock_hardware.gpu_info = [{'name': 'GeForce RTX 3080', 'total_memory_gb': 10.0}]
        mock_hardware.get_optimal_device.return_value = 'cuda:0'
        mock_hardware.get_optimal_model_config.return_value = {'name': 'medium', 'device': 'cuda:0'}
        mock_hardware.get_parallel_model_configs.return_value = [{'name': 'medium', 'device': 'cuda:0', 'batch_size': 4}]
        
        mock_get_hardware_info.return_value = mock_hardware
        
        mock_exists.return_value = False
        
        result = configure_models_for_hardware()
        
        mock_get_hardware_info.assert_called_once()
        
        self.assertIn('hardware_info', result)
        self.assertIn('optimal_device', result)
        self.assertIn('optimal_model', result)
        self.assertIn('available_models', result)
        self.assertIn('parallel_configs', result)
        
        self.assertEqual(result['hardware_info']['system'], {'system': 'Linux'})
        self.assertEqual(result['hardware_info']['cpu'], {'physical_cores': 4})
        self.assertEqual(result['hardware_info']['memory'], {'total_gb': 16.0})
        self.assertEqual(result['hardware_info']['gpu'], [{'name': 'GeForce RTX 3080', 'total_memory_gb': 10.0}])
        
        self.assertEqual(result['optimal_device'], 'cuda:0')
        
        self.assertEqual(result['optimal_model'], {'name': 'medium', 'device': 'cuda:0'})
        
        self.assertEqual(len(result['available_models']), 6)  # Default models
        
        self.assertEqual(result['parallel_configs']['llama-2-7b'], [{'name': 'medium', 'device': 'cuda:0', 'batch_size': 4}])
        self.assertEqual(result['parallel_configs']['llama-2-13b'], [{'name': 'medium', 'device': 'cuda:0', 'batch_size': 4}])
        self.assertEqual(result['parallel_configs']['llama-2-70b'], [{'name': 'medium', 'device': 'cuda:0', 'batch_size': 4}])
        self.assertEqual(result['parallel_configs']['mistral-7b'], [{'name': 'medium', 'device': 'cuda:0', 'batch_size': 4}])
        
        mock_exists.return_value = True
        mock_yaml_load.return_value = [
            {'name': 'custom', 'type': 'local', 'size_gb': 5.0, 'min_memory_gb': 6.0}
        ]
        
        result = configure_models_for_hardware(config_path='config.yaml')
        
        mock_yaml_load.assert_called_once()
        
        self.assertEqual(len(result['available_models']), 1)
        self.assertEqual(result['available_models'][0]['name'], 'custom')


if __name__ == '__main__':
    unittest.main()
