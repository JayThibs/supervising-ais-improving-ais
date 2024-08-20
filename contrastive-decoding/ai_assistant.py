from openai import OpenAI
import os

class AIAssistant:
    def __init__(self, model="gpt-4"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        if not self.client.api_key:
            raise ValueError("No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")

    def generate(self, prompt, max_tokens=150, temperature=0.7):
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in AI Assistant API call: {e}")
            return None

    def analyze(self, text, instruction):
        prompt = f"{instruction}\n\nText: {text}"
        return self.generate(prompt)