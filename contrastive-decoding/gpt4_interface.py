import openai
import os

class GPT4Interface:
    def __init__(self):
        # Load API key from environment variable
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")

    def generate(self, prompt, max_tokens=150, temperature=0.7):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in GPT-4 API call: {e}")
            return None

    def analyze(self, text, instruction):
        prompt = f"{instruction}\n\nText: {text}"
        return self.generate(prompt)