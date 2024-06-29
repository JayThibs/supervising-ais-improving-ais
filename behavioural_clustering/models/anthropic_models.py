import anthropic
from tenacity import retry, wait_random_exponential, stop_after_attempt

from models.model_factory import LanguageModelInterface

class AnthropicModel(LanguageModelInterface):
    def __init__(self, model, system_message, temperature=0.1, max_tokens=150):
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic()

    def generate(self, prompt):
        print("Generating with Anthropic API...")

        @retry(wait=wait_random_exponential(min=20, max=60), stop=stop_after_attempt(6))
        def completion_with_backoff(model, system_message, prompt):
            message = self.client.messages.create(
                model=model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_message,
                messages=[{"role": "user", "content": prompt}],
            )
            return message

        message = completion_with_backoff(
            model=self.model, system_message=self.system_message, prompt=prompt
        )

        print("Completed generation.")
        return message.content
