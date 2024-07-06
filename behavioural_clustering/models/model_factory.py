from api_models import AnthropicModel, OpenAIModel
from local_models import LocalModel

def initialize_model(model_info, temperature=0.1, max_tokens=150):
    """Initialize a language model."""
    model_family = model_info["model_family"]
    model = model_info["model"]
    system_message = model_info["system_message"]

    print("model_family:", model_family)
    print("model:", model)
    print("system_message:", system_message)
    
    model_class_map = {
        "openai": OpenAIModel,
        "anthropic": AnthropicModel,
        "local": LocalModel  # This should be replaced by Mistral and other models
    }
    
    if model_family in model_class_map:
        model_class = model_class_map[model_family]
        if model_family in ["openai", "anthropic"]:
            model_instance = model_class(model, system_message, temperature=temperature, max_tokens=max_tokens)
        else:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_instance = model_class(model, device=device)
            model_instance.load()
    else:
        raise ValueError(
            f"Invalid model family {model_family}. Options: 'openai', 'anthropic', 'local'."
        )
    
    return model_instance

class LanguageModelInterface:
    def generate(self, prompts):
        raise NotImplementedError("Subclasses should implement this method.")
