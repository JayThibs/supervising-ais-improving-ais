from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from behavioural_clustering.models.model_factory import LanguageModelInterface


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