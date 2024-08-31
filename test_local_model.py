import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from behavioural_clustering.models.local_models import LocalModel
from behavioural_clustering.utils.resource_management import ResourceManager
from dotenv import load_dotenv

# Print current working directory and Python path
print("Current working directory:", os.getcwd())
print("Python path:", sys.path)

# Set the project root to the current directory
project_root = os.getcwd()
sys.path.append(project_root)
print("Added to Python path:", project_root)

# Print contents of the src directory
src_dir = os.path.join(project_root, "src")
print("Contents of src directory:", os.listdir(src_dir))

# Attempt to import and print the result
try:
    from src.contrastive_decoding.contrastive_decoding import ContrastiveDecoder
    print("Import successful")
except ImportError as e:
    print("Import failed:", str(e))

load_dotenv()

# Now you can access the token like this:
hf_token = os.getenv("HUGGING_FACE_TOKEN")

def download_model(model_name):
    print(f"Downloading {model_name}...")
    AutoTokenizer.from_pretrained(model_name)
    AutoModelForCausalLM.from_pretrained(model_name)
    print(f"{model_name} downloaded successfully.")

def test_local_model(test_both=False, use_contrastive_decoding=False):
    model_names = ["google/gemma-2b-it", "google/gemma-2-2b-it"]
    model_names = [
        "HuggingFaceTB/SmolLM-135M-Instruct",
        "HuggingFaceTB/SmolLM-135M"
    ]
    
    if not test_both:
        model_names = model_names[:1]  # Only use the first model if not testing both
    
    # Download models if they're not already present
    for model_name in model_names:
        if not os.path.exists(os.path.join(os.getcwd(), model_name.split('/')[-1])):
            download_model(model_name)

    # Initialize and load the models
    local_models = [LocalModel(model_name) for model_name in model_names]

    # Check if there's enough space for the model(s)
    resource_manager = ResourceManager()
    device = resource_manager.get_available_devices()[0]  # Get the first available device
    total_memory = resource_manager.get_device_memory(device)
    
    model_memory_requirements = {
        model.model_name_or_path: model.get_memory_footprint() 
        for model in local_models
    }
    
    total_required_memory = sum(model_memory_requirements.values())
    
    print(f"Total available memory on {device}: {total_memory / 1e9:.2f} GB")
    print(f"Total required memory for model(s): {total_required_memory / 1e9:.2f} GB")
    
    if total_required_memory <= total_memory:
        print("There is enough space to fit the model(s).")
    else:
        print("There is not enough space to fit the model(s) simultaneously.")

    # Test generation with the model(s)
    prompt = "Write a short story about a robot learning to paint:"
    
    for model in local_models:
        print(f"\nTesting model: {model.model_name_or_path}")
        generated_text = model.generate(prompt)
        
        print(f"Prompt: {prompt}")
        print(f"Generated text: {generated_text}")

    if use_contrastive_decoding and len(local_models) == 2:
        print("\nTesting Contrastive Decoding:")
        cd = ContrastiveDecoder(
            model=local_models[0].model,
            comparison_model=local_models[1].model,
            tokenizer=local_models[0].tokenizer,
            device=device,
            generation_length=50,
            generations_per_prefix=1,
            starting_model_weight=1,
            comparison_model_weight=-1,
            single_prefix=prompt,
            return_divergences=True
        )
        result = cd.decode()
        print(f"Contrastive Decoding result: {result['texts'][0]}")
        print(f"Divergence: {result['divergences'][0]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test local model(s)")
    parser.add_argument("--test-both", action="store_true", help="Test both models instead of just one")
    parser.add_argument("--use-contrastive-decoding", action="store_true", help="Use contrastive decoding")
    args = parser.parse_args()

    test_local_model(test_both=args.test_both, use_contrastive_decoding=args.use_contrastive_decoding)