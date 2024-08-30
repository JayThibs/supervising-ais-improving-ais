import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LocalModel:
    def __init__(self, model_name_or_path, device="auto", max_length=150, temperature=0.01, top_p=0.9):
        self.model_name_or_path = model_name_or_path
        self.device = self._get_device(device)
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.model = None
        self.tokenizer = None
        self.load()  # Automatically load the model upon initialization

    def _get_device(self, device):
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)

    def load(self):
        print(f"Loading model: {self.model_name_or_path}")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        print(f"Model loaded successfully on device: {self.device}")

    def generate(self, prompt):
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load() method first.")

        print("Generating with local model...")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_new_tokens=self.max_length,
                    num_return_sequences=1
                )
        except NotImplementedError as e:
            if "MPS" in str(e):
                print("MPS device not supported for this operation. Falling back to CPU.")
                self.model = self.model.to("cpu")
                inputs = {k: v.to("cpu") for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        do_sample=True,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_new_tokens=self.max_length,
                        num_return_sequences=1
                    )
                self.model = self.model.to(self.device)
            else:
                raise e
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the input prompt from the generated text
        generated_text = generated_text[len(prompt):].strip()
        print("Completed generation.")
        return generated_text

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

    def get_memory_footprint(self):
        """
        Get the memory footprint of the model in bytes.
        """
        return sum(p.numel() * p.element_size() for p in self.model.parameters())
