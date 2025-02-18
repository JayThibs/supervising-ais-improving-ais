import os
import requests
import json
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
from tenacity import retry, wait_random_exponential, stop_after_attempt

load_dotenv()

class OpenAIModel:
    def __init__(self, model, system_message, temperature=0.01, max_tokens=150):
        self.model = model
        self.temperature = temperature
        self.default_max_tokens = max_tokens
        self.system_message = system_message
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate(self, prompt, max_tokens=None):
        @retry(wait=wait_random_exponential(min=20, max=60), stop=stop_after_attempt(6))
        def completion_with_backoff(model, prompt, max_tokens):
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
                max_tokens=max_tokens,
                timeout=60,
            )
            return completion

        max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        completion = completion_with_backoff(model=self.model, prompt=prompt, max_tokens=max_tokens)
        return completion.choices[0].message.content


class AnthropicModel:
    def __init__(self, model, system_message, temperature=0.01, max_tokens=150):
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.default_max_tokens = max_tokens
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def generate(self, prompt, max_tokens=None):
        print("Generating with Anthropic API...")

        @retry(wait=wait_random_exponential(min=20, max=60), stop=stop_after_attempt(6))
        def completion_with_backoff(model, system_message, prompt, max_tokens):
            message = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=self.temperature,
                system=system_message,
                messages=[{"role": "user", "content": prompt}],
            )
            return message

        max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        message = completion_with_backoff(
            model=self.model, system_message=self.system_message, prompt=prompt, max_tokens=max_tokens
        )
        print("Completed generation.")
        return message.content[0].text


class OpenRouterModel:
    def __init__(self, model, system_message, temperature=0.01, max_tokens=150):
        self.model = model
        self.temperature = temperature
        self.default_max_tokens = max_tokens
        self.system_message = system_message
        self.api_key = os.getenv("OPENROUTER_API_KEY")

    @retry(wait=wait_random_exponential(min=20, max=60), stop=stop_after_attempt(6))
    def generate(self, prompt, max_tokens=None):
        max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.temperature,
                "max_tokens": max_tokens
            }
        )
        
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
