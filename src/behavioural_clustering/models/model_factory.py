from .local_models import LocalModel

def initialize_model(model_info, temperature=0.1, max_tokens=150):
    model_family = model_info["model_family"]
    model = model_info["model"]
    system_message = model_info.get("system_message", "")

    if model_family == "openai":
        from .api_models import OpenAIModel
        return OpenAIModel(model, system_message, temperature=temperature, max_tokens=max_tokens)
    elif model_family == "anthropic":
        from .api_models import AnthropicModel
        return AnthropicModel(model, system_message, temperature=temperature, max_tokens=max_tokens)
    elif model_family == "local":
        return LocalModel(model, temperature=temperature, max_length=max_tokens)
    else:
        raise ValueError(f"Unsupported model family: {model_family}")
