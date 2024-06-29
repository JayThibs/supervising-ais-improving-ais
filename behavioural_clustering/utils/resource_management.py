import torch
import psutil
from typing import Dict, List

class ResourceManager:
    @staticmethod
    def get_available_devices() -> List[str]:
        """
        Get a list of available GPU devices.

        Returns:
            List[str]: List of available GPU device names.
        """
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else ["cpu"]

    @staticmethod
    def get_device_memory(device: str) -> int:
        """
        Get the available memory for a given device.

        Args:
            device (str): Device name (e.g., "cuda:0" or "cpu").

        Returns:
            int: Available memory in bytes.
        """
        return psutil.virtual_memory().available if device == "cpu" else torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)

    @staticmethod
    def allocate_models_to_devices(model_memory_requirements: Dict[str, int]) -> Dict[str, str]:
        """
        Allocate models to available devices based on memory requirements.

        Args:
            model_memory_requirements (Dict[str, int]): Dictionary of model names and their memory requirements.

        Returns:
            Dict[str, str]: Dictionary of model names and their allocated devices.
        """
        devices = ResourceManager.get_available_devices()
        allocation = {}
        for model, memory_req in model_memory_requirements.items():
            for device in devices:
                if ResourceManager.get_device_memory(device) >= memory_req:
                    allocation[model] = device
                    break
            else:
                raise ValueError(f"Not enough memory to allocate {model}")
        return allocation
    
    @staticmethod
    def check_gpu_availability() -> str:
        num_gpus = torch.cuda.device_count()
        return "multiple_gpus" if num_gpus > 1 else "single_gpu" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def check_gpu_memory(model_batch, buffer_factor=1.2) -> bool:
        if not torch.cuda.is_available():
            return False

        device = torch.device("cuda")
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        free_memory = total_memory - allocated_memory

        required_memory = sum(local_model.get_memory_usage() for _, local_model in model_batch)
        required_memory_with_buffer = required_memory * buffer_factor

        return free_memory >= required_memory_with_buffer