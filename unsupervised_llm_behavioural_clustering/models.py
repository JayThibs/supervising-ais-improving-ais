from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


class LanguageModelInterface:
    def generate(self, prompts):
        raise NotImplementedError("Subclasses should implement this method.")


class OpenAIModel(LanguageModelInterface):
    def __init__(self, model, system_message, temperature=0.1, max_tokens=150):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_message = system_message
        self.client = OpenAI()

    def generate(self, prompt):
        print("Generating with OpenAI API...")

        @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
        def completion_with_backoff(**kwargs):
            model = kwargs["model"]
            prompt = kwargs["prompt"]
            completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": self.system_message,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=10,
            )
            return completion

        completion = completion_with_backoff(model="gpt-3.5-turbo", prompt=prompt)

        print("Completed generation.")
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
