from openai import OpenAI
import anthropic
from tenacity import retry, wait_random_exponential, stop_after_attempt


class OpenAIModel:
    def __init__(self, model, system_message, temperature=0.1, max_tokens=150):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_message = system_message
        self.client = OpenAI()

    def generate(self, prompt):
        @retry(wait=wait_random_exponential(min=20, max=60), stop=stop_after_attempt(6))
        def completion_with_backoff(model, prompt):
            completion = self.client.chat.completions.create(
                model=model,
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
                timeout=60,
            )
            return completion

        completion = completion_with_backoff(model=self.model, prompt=prompt)
        return completion.choices[0].message.content


class AnthropicModel:
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
