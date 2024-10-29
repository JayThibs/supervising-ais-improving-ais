import torch
import logging
import platform

logger = logging.getLogger(__name__)

def get_device(requested_device: str = None) -> str:
    """
    Get the appropriate device, prioritizing CUDA if available.
    
    Args:
        requested_device: Device requested ('cuda', 'mps', 'cpu', or None)
    
    Returns:
        str: Available device ('cuda', 'mps', or 'cpu')
    """
    # Log system info for debugging
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    
    if requested_device == 'cpu':
        logger.info("CPU explicitly requested")
        return 'cpu'

    # First try CUDA
    if torch.cuda.is_available():
        logger.info("CUDA device available and selected")
        return 'cuda'
        
    # Then try MPS (Apple Silicon)
    if platform.processor() == 'arm' and platform.system() == 'Darwin':
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            logger.info("MPS (Apple Silicon) device selected")
            return 'mps'
    
    # Fallback to CPU
    logger.info("No GPU available, defaulting to CPU device")
    return 'cpu'
