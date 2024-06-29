from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt

from models.model_factory import LanguageModelInterface


class OpenAIModel(LanguageModelInterface):
    def __init__(self, model, system_message, temperature=0.1, max_tokens=150):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_message = system_message
        self.client = OpenAI()

    def generate(self, prompt):
        print("Generating with OpenAI API...")

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

        completion = completion_with_backoff(model="gpt-3.5-turbo", prompt=prompt)

        print("Completed generation.")
        return completion.choices[0].message.content