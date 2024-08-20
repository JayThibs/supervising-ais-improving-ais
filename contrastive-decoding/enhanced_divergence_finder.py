import json
import re
from typing import List
from ai_assistant import AIAssistant

class EnhancedDivergenceFinder:
    def __init__(self, model1, model2, topics, ai_model="gpt-4"):
        self.model1 = model1
        self.model2 = model2
        self.topics = topics
        self.ai_assistant = AIAssistant(model=ai_model)
        self.contrastive_decoder = ContrastiveDecoder(model1=model1, model2=model2, experiment_type='targeted', targeted_topics=topics)

    def generate_prompts(self, n_prompts=10):
        context = f"Generate {n_prompts} prompts to test differences between two language models on the following topics: {', '.join(self.topics)}. Focus on potential areas of divergence."
        response = self.ai_assistant.generate(context)
        return self.parse_ai_response(response)

    def analyze_divergences(self, divergences):
        context = f"Analyze the following divergences between two language models: {divergences}. Identify significant differences and potential implications."
        return self.ai_assistant.generate(context)

    def find_divergences(self, prompts):
        results = self.contrastive_decoder.decode(text_set=prompts)
        return results['targeted_divergences']

    def parse_ai_response(self, response: str) -> List[str]:
        # Try to parse as JSON first
        try:
            json_response = json.loads(response)
            if isinstance(json_response, dict):
                # Extract prompts from dictionary
                prompts = [v for k, v in json_response.items() if 'prompt' in k.lower()]
            elif isinstance(json_response, list):
                # If it's a list, assume each item is a prompt
                prompts = json_response
            else:
                raise ValueError("Unexpected JSON structure")
        except json.JSONDecodeError:
            # If not JSON, treat as plain text
            prompts = self._parse_plain_text(response)

        # Clean and validate prompts
        cleaned_prompts = [self._clean_prompt(p) for p in prompts if p]
        return cleaned_prompts

    def _parse_plain_text(self, text: str) -> List[str]:
        # Split by common delimiters
        lines = re.split(r'\n|\.|\d+\)', text)
        # Remove empty lines and strip whitespace
        return [line.strip() for line in lines if line.strip()]

    def _clean_prompt(self, prompt: str) -> str:
        # Remove any leading numbers or special characters
        prompt = re.sub(r'^[\d\W]+', '', prompt).strip()
        # Remove any "Prompt:" or similar prefixes
        prompt = re.sub(r'^(prompt|question|query):\s*', '', prompt, flags=re.IGNORECASE)
        return prompt