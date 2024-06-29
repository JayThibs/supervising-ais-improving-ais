from models.openai_models import OpenAIModel
from models.anthropic_models import AnthropicModel
from models.local_models import LocalModel


def initialize_model(model_info, temperature=0.1, max_tokens=150):
    """Initialize a language model."""
    model_family, model, system_message = (
        model_info["model_family"],
        model_info["model"],
        model_info["system_message"],
    )
    print("model_family:", model_family)
    print("model:", model)
    print("system_message:", system_message)
    if model_family == "openai":
        model_instance = OpenAIModel(
            model, system_message, temperature=temperature, max_tokens=max_tokens
        )
    elif model_family == "anthropic":
        model_instance = AnthropicModel(model)
    elif model_family == "local":  # This should be replaced by Mistral and other models
        model_instance = LocalModel(model)
    else:
        raise ValueError(
            f"Invalid model family {model_family}. Options: 'openai', 'anthropic', 'local'."
        )
    return model_instance


class LanguageModelInterface:
    def generate(self, prompts):
        raise NotImplementedError("Subclasses should implement this method.")