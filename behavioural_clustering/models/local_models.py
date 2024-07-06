from transformers import AutoModelForCausalLM, AutoTokenizer
from model_factory import LanguageModelInterface
import torch

class LocalModel(LanguageModelInterface):
    """Class for local Hugging Face models compatible with contrastive decoding."""

    def __init__(self, model_name_or_path, device="auto", max_length=150, temperature=0.1, top_p=0.9):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.model = None
        self.tokenizer = None

    def load(self):
        print(f"Loading model: {self.model_name_or_path}")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        print("Model loaded successfully.")

    def generate(self, prompt):
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load() method first.")

        print("Generating with local model...")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                max_length=self.max_length,
                num_return_sequences=1
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Completed generation.")
        return generated_text

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer
