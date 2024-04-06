from openai import OpenAI
import anthropic
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


from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class LocalModel(LanguageModelInterface):
    """Superclass for local models."""

    def __init__(self, model_name_or_path, device=-1, max_length=150):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        self.pipeline = None

    def load(self):
        print(f"Loading model: {self.model_name_or_path}")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            max_length=self.max_length,
        )
        print("Model loaded successfully.")

    def generate(self, prompt):
        if self.pipeline is None:
            raise ValueError("Model not loaded. Call load() method first.")

        print("Generating with local model...")
        output = self.pipeline(prompt, num_return_sequences=1)
        generated_text = output[0]["generated_text"]
        print("Completed generation.")
        return generated_text
