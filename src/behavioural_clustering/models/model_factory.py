from .local_models import LocalModel
from .api_models import OpenRouterModel

def initialize_model(model_info, temperature=0.01, max_tokens=150, device="auto"):
    model_family = model_info["model_family"]
    model_name = model_info["model_name"]
    system_message = model_info.get("system_message", "")

    if model_family == "openai":
        from .api_models import OpenAIModel
        return OpenAIModel(model_name, system_message, temperature=temperature, max_tokens=max_tokens)
    elif model_family == "anthropic":
        from .api_models import AnthropicModel
        return AnthropicModel(model_name, system_message, temperature=temperature, max_tokens=max_tokens)
    elif model_family == "local":
        return LocalModel(model_name, device=device, temperature=temperature, max_length=max_tokens)
    elif model_family == "openrouter":
        return OpenRouterModel(model_name, system_message, temperature=temperature, max_tokens=max_tokens)
    elif model_family == "huggingface":
        from .huggingface_models import HuggingfaceModelAdapter
        return HuggingfaceModelAdapter(model_name, system_message, device=device, temperature=temperature, max_tokens=max_tokens)
    else:
        raise ValueError(f"Unsupported model family: {model_family}")
