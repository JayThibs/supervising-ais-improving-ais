import os
import platform
import logging
import subprocess
from typing import Dict, Any, List, Optional, Tuple
import psutil
import torch
from pathlib import Path

logger = logging.getLogger(__name__)

class HardwareInfo:
    """Class for detecting and managing hardware resources for LLM usage."""
    
    def __init__(self):
        """Initialize hardware detection."""
        self.system_info = self._get_system_info()
        self.cpu_info = self._get_cpu_info()
        self.memory_info = self._get_memory_info()
        self.gpu_info = self._get_gpu_info()
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get basic system information."""
        return {
            'platform': platform.platform(),
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor()
        }
        
    def _get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU information."""
        return {
            'physical_cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(logical=True),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {}
        }
        
    def _get_memory_info(self) -> Dict[str, Any]:
        """Get memory information."""
        mem = psutil.virtual_memory()
        return {
            'total': mem.total,
            'available': mem.available,
            'used': mem.used,
            'percent': mem.percent,
            'total_gb': mem.total / (1024 ** 3),
            'available_gb': mem.available / (1024 ** 3)
        }
        
    def _get_gpu_info(self) -> List[Dict[str, Any]]:
        """Get GPU information if available."""
        gpus = []
        
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            
            for i in range(device_count):
                device = torch.cuda.get_device_properties(i)
                gpus.append({
                    'index': i,
                    'name': device.name,
                    'total_memory': device.total_memory,
                    'total_memory_gb': device.total_memory / (1024 ** 3),
                    'cuda_version': cuda_version,
                    'compute_capability': f"{device.major}.{device.minor}"
                })
                
        return gpus
        
    def get_optimal_device(self) -> str:
        """
        Get the optimal device for running models.
        
        Returns:
            Device string ('cuda:0', 'cuda:1', etc. or 'cpu')
        """
        if torch.cuda.is_available():
            best_gpu = 0
            max_memory = 0
            
            for i in range(torch.cuda.device_count()):
                try:
                    torch.cuda.set_device(i)
                    torch.cuda.empty_cache()
                    free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
                    
                    if free_memory > max_memory:
                        max_memory = free_memory
                        best_gpu = i
                except Exception as e:
                    logger.warning(f"Error checking GPU {i}: {str(e)}")
                    continue
                    
            return f"cuda:{best_gpu}"
        else:
            return "cpu"
            
    def get_optimal_batch_size(self, model_size_gb: float = 7.0, device: Optional[str] = None) -> int:
        """
        Calculate optimal batch size based on available memory.
        
        Args:
            model_size_gb: Approximate size of the model in GB
            device: Device to check (if None, uses optimal device)
            
        Returns:
            Recommended batch size
        """
        if device is None:
            device = self.get_optimal_device()
            
        if device.startswith("cuda"):
            gpu_idx = int(device.split(":")[-1])
            
            if gpu_idx < len(self.gpu_info):
                gpu = self.gpu_info[gpu_idx]
                total_memory_gb = gpu['total_memory_gb']
                
                usable_memory_gb = total_memory_gb * 0.7
                
                remaining_memory_gb = usable_memory_gb - model_size_gb
                
                batch_size = max(1, int(remaining_memory_gb / 0.5))
                
                return min(batch_size, 32)  # Cap at 32 to avoid excessive memory usage
            else:
                return 1  # Default to 1 if GPU info not available
        else:
            available_gb = self.memory_info['available_gb']
            
            usable_memory_gb = available_gb * 0.5
            
            remaining_memory_gb = usable_memory_gb - model_size_gb
            
            batch_size = max(1, int(remaining_memory_gb / 0.5))
            
            return min(batch_size, 8)  # Cap at 8 for CPU to avoid excessive memory usage
            
    def get_optimal_model_config(self, available_models: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Determine the optimal model configuration based on hardware.
        
        Args:
            available_models: List of model configurations with 'name', 'size_gb', and 'min_memory_gb' keys
            
        Returns:
            Optimal model configuration
        """
        device = self.get_optimal_device()
        
        if device.startswith("cuda"):
            gpu_idx = int(device.split(":")[-1])
            
            if gpu_idx < len(self.gpu_info):
                gpu = self.gpu_info[gpu_idx]
                total_memory_gb = gpu['total_memory_gb']
                
                for model in sorted(available_models, key=lambda m: m.get('size_gb', 0), reverse=True):
                    if model.get('min_memory_gb', 0) <= total_memory_gb * 0.7:
                        return {**model, 'device': device}
                        
                return {**available_models[0], 'device': device}
            else:
                return {**available_models[0], 'device': device}
        else:
            available_gb = self.memory_info['available_gb']
            
            for model in sorted(available_models, key=lambda m: m.get('size_gb', 0), reverse=True):
                if model.get('min_memory_gb', 0) <= available_gb * 0.5:
                    return {**model, 'device': device}
                    
            return {**available_models[0], 'device': device}
            
    def get_parallel_model_configs(self, available_models: List[Dict[str, Any]], 
                                 model_name: str) -> List[Dict[str, Any]]:
        """
        Determine configurations for running multiple instances of a model in parallel.
        
        Args:
            available_models: List of model configurations
            model_name: Name of the model to run in parallel
            
        Returns:
            List of model configurations for parallel execution
        """
        model_config = None
        for model in available_models:
            if model.get('name') == model_name:
                model_config = model
                break
                
        if not model_config:
            logger.warning(f"Model {model_name} not found in available models")
            return []
            
        model_size_gb = model_config.get('size_gb', 7.0)
        
        if torch.cuda.is_available():
            configs = []
            
            for i in range(torch.cuda.device_count()):
                gpu = self.gpu_info[i]
                total_memory_gb = gpu['total_memory_gb']
                
                if model_size_gb <= total_memory_gb * 0.7:
                    configs.append({
                        **model_config,
                        'device': f"cuda:{i}",
                        'batch_size': self.get_optimal_batch_size(model_size_gb, f"cuda:{i}")
                    })
                    
            return configs
        else:
            return [{
                **model_config,
                'device': 'cpu',
                'batch_size': self.get_optimal_batch_size(model_size_gb, 'cpu')
            }]
            
    def print_hardware_summary(self):
        """Print a summary of detected hardware."""
        print("\n=== Hardware Summary ===")
        print(f"System: {self.system_info['system']} {self.system_info['release']}")
        print(f"CPU: {self.cpu_info['physical_cores']} physical cores, {self.cpu_info['logical_cores']} logical cores")
        print(f"Memory: {self.memory_info['total_gb']:.1f} GB total, {self.memory_info['available_gb']:.1f} GB available")
        
        if self.gpu_info:
            print("\nGPUs:")
            for gpu in self.gpu_info:
                print(f"  {gpu['name']} - {gpu['total_memory_gb']:.1f} GB")
                
            print(f"\nOptimal device: {self.get_optimal_device()}")
        else:
            print("\nNo GPUs detected. Using CPU.")
            
        print("=====================\n")


def get_hardware_info() -> HardwareInfo:
    """
    Get hardware information.
    
    Returns:
        HardwareInfo instance
    """
    return HardwareInfo()


def configure_models_for_hardware(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Configure models based on available hardware.
    
    Args:
        config_path: Path to model configuration file
        
    Returns:
        Dictionary of model configurations
    """
    hardware = get_hardware_info()
    
    default_models = [
        {
            'name': 'gpt-3.5-turbo',
            'type': 'api',
            'provider': 'openai',
            'size_gb': 0,  # API model doesn't use local memory
            'min_memory_gb': 0
        },
        {
            'name': 'claude-3-opus-20240229',
            'type': 'api',
            'provider': 'anthropic',
            'size_gb': 0,  # API model doesn't use local memory
            'min_memory_gb': 0
        },
        {
            'name': 'llama-2-7b',
            'type': 'local',
            'size_gb': 7.0,
            'min_memory_gb': 10.0
        },
        {
            'name': 'llama-2-13b',
            'type': 'local',
            'size_gb': 13.0,
            'min_memory_gb': 16.0
        },
        {
            'name': 'llama-2-70b',
            'type': 'local',
            'size_gb': 70.0,
            'min_memory_gb': 80.0
        },
        {
            'name': 'mistral-7b',
            'type': 'local',
            'size_gb': 7.0,
            'min_memory_gb': 10.0
        }
    ]
    
    models = default_models
    if config_path and config_path.exists():
        try:
            import yaml
            with open(config_path, 'r') as f:
                custom_models = yaml.safe_load(f)
                if custom_models and isinstance(custom_models, list):
                    models = custom_models
        except Exception as e:
            logger.warning(f"Error loading model configuration: {str(e)}")
    
    result = {
        'hardware_info': {
            'system': hardware.system_info,
            'cpu': hardware.cpu_info,
            'memory': hardware.memory_info,
            'gpu': hardware.gpu_info
        },
        'optimal_device': hardware.get_optimal_device(),
        'optimal_model': hardware.get_optimal_model_config(models),
        'available_models': models,
        'parallel_configs': {}
    }
    
    for model in models:
        if model.get('type') == 'local':
            result['parallel_configs'][model['name']] = hardware.get_parallel_model_configs(models, model['name'])
    
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    hardware = get_hardware_info()
    hardware.print_hardware_summary()
    
    model_configs = configure_models_for_hardware()
    
    optimal_model = model_configs['optimal_model']
    print(f"Optimal model: {optimal_model['name']} on {optimal_model['device']}")
    
    print("\nParallel configurations:")
    for model_name, configs in model_configs['parallel_configs'].items():
        print(f"  {model_name}: {len(configs)} instances")
