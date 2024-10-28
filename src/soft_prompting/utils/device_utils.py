import torch

def get_device(requested_device: str = "cuda") -> str:
    """
    Determine the appropriate device based on availability.
    
    Args:
        requested_device: Desired device ("cuda", "mps", or "cpu")
        
    Returns:
        str: Available device ("cuda", "mps", or "cpu")
    """
    if requested_device == "cuda" and torch.cuda.is_available():
        return "cuda"
    elif requested_device == "mps" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
