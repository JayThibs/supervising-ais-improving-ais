from openai import OpenAI


class LanguageModelInterface:
    def generate(self, prompts):
        raise NotImplementedError("Subclasses should implement this method.")


class OpenAIModel(LanguageModelInterface):
    def __init__(self, model, temperature=0.5, max_tokens=150):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt, system_message="You are an AI language model."):
        client = OpenAI()

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return completion.choices[0].message.content


class AnthropicModel(LanguageModelInterface):
    def __init__(self, model):
        self.model = model

    def generate(self, prompts):
        # Your Anthropic API logic here
        pass


class LocalModel(LanguageModelInterface):
    """Superclass for local models."""

    def __init__(self, model):
        # Your local model setup logic here
        self.model = model

    def load(self):
        # Your local model loading logic here
        pass

    def generate(self, prompts):
        # Your local model logic here
        pass
