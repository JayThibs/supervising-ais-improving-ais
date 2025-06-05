"""
Model Utilities Module

This module provides utility functions for working with language models in the
mechanistic discovery pipeline. It handles model loading, compatibility checking,
and configuration management.

Key Features:
    - Model compatibility verification
    - Memory-efficient model loading
    - Model metadata extraction
    - Configuration validation
"""

import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Dict, Any, Optional, Tuple, List
import gc
from pathlib import Path
import json
import psutil
import warnings


def get_model_info(model_name_or_path: str) -> Dict[str, Any]:
    """
    Extract detailed information about a model without fully loading it.
    
    Args:
        model_name_or_path: HuggingFace model name or local path
        
    Returns:
        Dictionary containing model information
    """
    try:
        # Load config without loading weights
        config = AutoConfig.from_pretrained(model_name_or_path)
        
        info = {
            'model_type': config.model_type,
            'architecture': config.architectures[0] if config.architectures else 'unknown',
            'n_layers': getattr(config, 'num_hidden_layers', 
                              getattr(config, 'n_layers', 
                                    getattr(config, 'n_layer', None))),
            'hidden_size': getattr(config, 'hidden_size', 
                                 getattr(config, 'd_model', 
                                       getattr(config, 'n_embd', None))),
            'intermediate_size': getattr(config, 'intermediate_size',
                                       getattr(config, 'mlp_dim',
                                             getattr(config, 'n_inner', None))),
            'n_heads': getattr(config, 'num_attention_heads',
                             getattr(config, 'n_heads',
                                   getattr(config, 'n_head', None))),
            'vocab_size': config.vocab_size,
            'max_position_embeddings': getattr(config, 'max_position_embeddings',
                                             getattr(config, 'n_positions',
                                                   getattr(config, 'n_ctx', None))),
            'torch_dtype': str(getattr(config, 'torch_dtype', 'float32')),
        }
        
        # Estimate model size
        param_count = estimate_model_parameters(info)
        info['estimated_parameters'] = param_count
        info['estimated_size_gb'] = param_count * 4 / (1024**3)  # Assuming float32
        
        # Check for special features
        info['has_transcoders'] = check_transcoder_availability(model_name_or_path)
        info['supports_circuit_tracing'] = check_circuit_tracing_support(info['architecture'])
        
        return info
        
    except Exception as e:
        warnings.warn(f"Could not load model info for {model_name_or_path}: {e}")
        return {
            'error': str(e),
            'model_type': 'unknown',
            'architecture': 'unknown'
        }


def check_model_compatibility(base_model_name: str, 
                            intervention_model_name: str) -> Tuple[bool, List[str]]:
    """
    Check if two models are compatible for circuit comparison.
    
    Args:
        base_model_name: Name/path of base model
        intervention_model_name: Name/path of intervention model
        
    Returns:
        Tuple of (is_compatible, list_of_issues)
    """
    issues = []
    
    # Get model info
    base_info = get_model_info(base_model_name)
    int_info = get_model_info(intervention_model_name)
    
    # Check for errors
    if 'error' in base_info:
        issues.append(f"Could not load base model: {base_info['error']}")
        return False, issues
        
    if 'error' in int_info:
        issues.append(f"Could not load intervention model: {int_info['error']}")
        return False, issues
        
    # Check architecture compatibility
    if base_info['architecture'] != int_info['architecture']:
        issues.append(
            f"Architecture mismatch: {base_info['architecture']} vs {int_info['architecture']}"
        )
        
    # Check dimensions
    if base_info['n_layers'] != int_info['n_layers']:
        issues.append(
            f"Layer count mismatch: {base_info['n_layers']} vs {int_info['n_layers']}"
        )
        
    if base_info['hidden_size'] != int_info['hidden_size']:
        issues.append(
            f"Hidden size mismatch: {base_info['hidden_size']} vs {int_info['hidden_size']}"
        )
        
    if base_info['intermediate_size'] != int_info['intermediate_size']:
        issues.append(
            f"Intermediate size mismatch: {base_info['intermediate_size']} vs {int_info['intermediate_size']}"
        )
        
    # Check tokenizer compatibility
    tokenizer_compatible, tokenizer_issues = check_tokenizer_compatibility(
        base_model_name, intervention_model_name
    )
    issues.extend(tokenizer_issues)
    
    # Check circuit tracing support
    if not base_info.get('supports_circuit_tracing', False):
        issues.append(f"Base model {base_info['architecture']} may not support circuit tracing")
        
    if not int_info.get('supports_circuit_tracing', False):
        issues.append(f"Intervention model {int_info['architecture']} may not support circuit tracing")
        
    # Determine if compatible despite issues
    critical_issues = [
        issue for issue in issues 
        if 'mismatch' in issue and ('layer' in issue or 'size' in issue)
    ]
    
    is_compatible = len(critical_issues) == 0
    
    return is_compatible, issues


def check_tokenizer_compatibility(model1_name: str, model2_name: str) -> Tuple[bool, List[str]]:
    """
    Check if two models use compatible tokenizers.
    
    Args:
        model1_name: First model name/path
        model2_name: Second model name/path
        
    Returns:
        Tuple of (is_compatible, list_of_issues)
    """
    issues = []
    
    try:
        tokenizer1 = AutoTokenizer.from_pretrained(model1_name)
        tokenizer2 = AutoTokenizer.from_pretrained(model2_name)
        
        # Check vocab size
        if tokenizer1.vocab_size != tokenizer2.vocab_size:
            issues.append(
                f"Vocabulary size mismatch: {tokenizer1.vocab_size} vs {tokenizer2.vocab_size}"
            )
            
        # Check special tokens
        special_tokens1 = set(tokenizer1.all_special_tokens)
        special_tokens2 = set(tokenizer2.all_special_tokens)
        
        if special_tokens1 != special_tokens2:
            diff = special_tokens1.symmetric_difference(special_tokens2)
            issues.append(f"Special token mismatch: {diff}")
            
        # Test encoding consistency
        test_text = "Hello, this is a test."
        tokens1 = tokenizer1.encode(test_text)
        tokens2 = tokenizer2.encode(test_text)
        
        if tokens1 != tokens2:
            issues.append("Tokenizers produce different encodings for the same text")
            
    except Exception as e:
        issues.append(f"Error checking tokenizer compatibility: {e}")
        
    is_compatible = len(issues) == 0
    return is_compatible, issues


def estimate_model_parameters(model_info: Dict[str, Any]) -> int:
    """
    Estimate the number of parameters in a model based on its configuration.
    
    Args:
        model_info: Dictionary with model configuration
        
    Returns:
        Estimated parameter count
    """
    n_layers = model_info.get('n_layers', 12)
    hidden_size = model_info.get('hidden_size', 768)
    intermediate_size = model_info.get('intermediate_size', hidden_size * 4)
    vocab_size = model_info.get('vocab_size', 50000)
    
    # Embedding parameters
    embedding_params = vocab_size * hidden_size + hidden_size * 512  # token + position
    
    # Attention parameters per layer
    attention_params = 4 * hidden_size * hidden_size  # Q, K, V, O projections
    
    # MLP parameters per layer
    mlp_params = 2 * hidden_size * intermediate_size  # Up and down projections
    
    # Layer norm parameters
    ln_params = 4 * hidden_size * n_layers  # 2 per layer + 2 global
    
    # Total
    total_params = (
        embedding_params +
        n_layers * (attention_params + mlp_params) +
        ln_params
    )
    
    return int(total_params)


def check_memory_requirements(model_info: Dict[str, Any], 
                            batch_size: int = 8,
                            sequence_length: int = 512) -> Dict[str, float]:
    """
    Estimate memory requirements for running the model.
    
    Args:
        model_info: Model information dictionary
        batch_size: Batch size for inference
        sequence_length: Maximum sequence length
        
    Returns:
        Dictionary with memory estimates in GB
    """
    # Model parameters
    param_memory = model_info.get('estimated_size_gb', 1.0)
    
    # Activation memory (rough estimate)
    n_layers = model_info.get('n_layers', 12)
    hidden_size = model_info.get('hidden_size', 768)
    
    # Memory per token in batch
    memory_per_token = (
        hidden_size * 4 +  # Hidden states
        hidden_size * n_layers * 2  # Intermediate activations
    ) * 4 / (1024**3)  # Convert to GB
    
    activation_memory = memory_per_token * batch_size * sequence_length
    
    # Gradient memory (if training)
    gradient_memory = param_memory
    
    # Circuit tracing overhead (estimated)
    circuit_overhead = param_memory * 0.5  # Transcoders and graphs
    
    return {
        'model_parameters_gb': param_memory,
        'activation_memory_gb': activation_memory,
        'gradient_memory_gb': gradient_memory,
        'circuit_tracing_overhead_gb': circuit_overhead,
        'total_inference_gb': param_memory + activation_memory + circuit_overhead,
        'total_training_gb': param_memory + activation_memory + gradient_memory + circuit_overhead
    }


def get_available_memory() -> Dict[str, float]:
    """
    Get available system and GPU memory.
    
    Returns:
        Dictionary with memory information in GB
    """
    # System memory
    vm = psutil.virtual_memory()
    system_memory = {
        'total_gb': vm.total / (1024**3),
        'available_gb': vm.available / (1024**3),
        'used_gb': vm.used / (1024**3),
        'percent_used': vm.percent
    }
    
    # GPU memory
    gpu_memory = {'available': False}
    
    if torch.cuda.is_available():
        try:
            gpu_memory['available'] = True
            gpu_memory['device_count'] = torch.cuda.device_count()
            
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_device(i)
                total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                
                gpu_memory[f'gpu_{i}'] = {
                    'total_gb': total,
                    'allocated_gb': allocated,
                    'reserved_gb': reserved,
                    'available_gb': total - reserved
                }
        except Exception as e:
            gpu_memory['error'] = str(e)
            
    return {
        'system': system_memory,
        'gpu': gpu_memory
    }


def check_transcoder_availability(model_name_or_path: str) -> bool:
    """
    Check if transcoders are available for a model.
    
    Args:
        model_name_or_path: Model name or path
        
    Returns:
        True if transcoders are available
    """
    # Known models with pre-trained transcoders
    models_with_transcoders = [
        'gpt2',
        'gpt2-medium',
        'gpt2-large',
        'gpt2-xl',
        'EleutherAI/pythia-70m',
        'EleutherAI/pythia-160m',
        'EleutherAI/pythia-410m',
        'EleutherAI/pythia-1b',
        'EleutherAI/pythia-1.4b',
        'EleutherAI/pythia-2.8b',
        'EleutherAI/pythia-6.9b',
        'EleutherAI/pythia-12b'
    ]
    
    # Check if model is in known list
    for known_model in models_with_transcoders:
        if known_model in model_name_or_path:
            return True
            
    # Check for local transcoder files
    if Path(model_name_or_path).exists():
        transcoder_dir = Path(model_name_or_path) / 'transcoders'
        if transcoder_dir.exists():
            return True
            
    # Check for separate transcoder directory
    transcoder_path = Path(f"{model_name_or_path}_transcoders")
    if transcoder_path.exists():
        return True
        
    return False


def check_circuit_tracing_support(architecture: str) -> bool:
    """
    Check if an architecture supports circuit tracing.
    
    Args:
        architecture: Model architecture name
        
    Returns:
        True if architecture is supported
    """
    supported_architectures = [
        'GPT2LMHeadModel',
        'GPT2Model', 
        'GPTNeoForCausalLM',
        'GPTNeoModel',
        'GPTJForCausalLM',
        'GPTJModel',
        'LlamaForCausalLM',
        'LlamaModel',
        'MistralForCausalLM',
        'MistralModel',
        'Qwen2ForCausalLM',
        'Qwen2Model'
    ]
    
    return any(arch in architecture for arch in supported_architectures)


def optimize_model_loading(model_name: str,
                         device: str = 'cuda',
                         quantization: Optional[str] = None) -> Dict[str, Any]:
    """
    Get optimal loading configuration for a model.
    
    Args:
        model_name: Model name or path
        device: Target device
        quantization: Quantization method ('8bit', '4bit', None)
        
    Returns:
        Dictionary with loading configuration
    """
    # Get model info
    model_info = get_model_info(model_name)
    
    # Get available memory
    memory_info = get_available_memory()
    
    # Determine optimal configuration
    config = {
        'device_map': 'auto',
        'torch_dtype': torch.float16,
        'low_cpu_mem_usage': True
    }
    
    # Check if model fits in GPU memory
    if device == 'cuda' and memory_info['gpu']['available']:
        gpu_memory = memory_info['gpu']['gpu_0']['available_gb']
        model_size = model_info.get('estimated_size_gb', 1.0)
        
        if model_size > gpu_memory * 0.8:  # Leave some headroom
            if quantization is None:
                # Suggest quantization
                if model_size > gpu_memory * 1.5:
                    config['load_in_4bit'] = True
                    config['bnb_4bit_compute_dtype'] = torch.float16
                    config['bnb_4bit_use_double_quant'] = True
                else:
                    config['load_in_8bit'] = True
            else:
                # Use specified quantization
                if quantization == '4bit':
                    config['load_in_4bit'] = True
                    config['bnb_4bit_compute_dtype'] = torch.float16
                elif quantization == '8bit':
                    config['load_in_8bit'] = True
                    
    return config


def clear_model_cache():
    """
    Clear model caches to free memory.
    
    This is useful between loading different models.
    """
    # Clear PyTorch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
    # Force garbage collection
    gc.collect()
    
    # Clear transformers cache (if needed)
    import transformers
    if hasattr(transformers, 'modeling_utils'):
        if hasattr(transformers.modeling_utils, '_model_cache'):
            transformers.modeling_utils._model_cache.clear()


def validate_circuit_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate a circuit tracing configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    required_fields = ['model_name', 'transcoder_set']
    
    # Check required fields
    for field in required_fields:
        if field not in config:
            issues.append(f"Missing required field: {field}")
            
    # Validate model
    if 'model_name' in config:
        model_info = get_model_info(config['model_name'])
        if 'error' in model_info:
            issues.append(f"Cannot load model: {model_info['error']}")
            
    # Validate transcoder availability
    if 'transcoder_set' in config and 'model_name' in config:
        if not check_transcoder_availability(config['transcoder_set']):
            issues.append(f"Transcoders not found: {config['transcoder_set']}")
            
    # Validate device
    if 'device' in config:
        if config['device'] == 'cuda' and not torch.cuda.is_available():
            issues.append("CUDA requested but not available")
            
    # Validate numerical parameters
    if 'batch_size' in config and config['batch_size'] <= 0:
        issues.append("Batch size must be positive")
        
    is_valid = len(issues) == 0
    return is_valid, issues