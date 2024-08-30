import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from behavioural_clustering.models.local_models import LocalModel
from behavioural_clustering.utils.resource_management import ResourceManager

def download_model(model_name):
    print(f"Downloading {model_name}...")
    AutoTokenizer.from_pretrained(model_name)
    AutoModelForCausalLM.from_pretrained(model_name)
    print(f"{model_name} downloaded successfully.")

def test_local_model(test_both=False):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test local model(s)")
    parser.add_argument("--test-both", action="store_true", help="Test both models instead of just one")
    args = parser.parse_args()

    test_local_model(test_both=args.test_both)