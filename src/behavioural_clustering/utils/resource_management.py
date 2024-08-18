import torch
import psutil
from typing import Dict, List

class ResourceManager:
    @staticmethod
    def get_available_devices() -> List[str]:
        """
        Get a list of available GPU devices including support for Apple Silicon (MPS).

        Returns:
            List[str]: List of available GPU device names.
        """
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
        if torch.backends.mps.is_available():
            devices.append("mps")
        if not devices:  # If no GPU or MPS devices, return CPU
            devices.append("cpu")
        return devices

    @staticmethod
    def get_device_memory(device: str) -> int:
        """
        Get the available memory for a given device.

        Args:
            device (str): Device name (e.g., "cuda:0", "mps" or "cpu").

        Returns:
            int: Available memory in bytes.
        """
        if device == "cpu":
            return psutil.virtual_memory().available
        elif device == "mps":
            # Assuming a fixed amount of memory for MPS devices as MPS does not expose memory management APIs
            return 8 * 1024**3  # 8 GB as a placeholder
        else:
            return torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)

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
        has_mps = torch.backends.mps.is_available()
        if num_gpus > 1:
            return "multiple_gpus"
        elif num_gpus == 1:
            return "single_gpu"
        elif has_mps:
            return "mps_available"
        else:
            return "cpu"

    @staticmethod
    def check_gpu_memory(model_batch, buffer_factor=1.2) -> bool:
        if not torch.cuda.is_available() and not torch.backends.mps.is_available():
            return False

        device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        if device.type == "cuda":
            total_memory = torch.cuda.get_device_properties(device).total_memory
            allocated_memory = torch.cuda.memory_allocated(device)
            free_memory = total_memory - allocated_memory
        else:
            # Assuming a fixed free memory for MPS devices
            free_memory = 8 * 1024**3  # 8 GB as a placeholder

        required_memory = sum(local_model.get_memory_usage() for _, local_model in model_batch)
        required_memory_with_buffer = required_memory * buffer_factor

        return free_memory >= required_memory_with_buffer