import argparse
import pandas as pd
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from auto_finetuning_data import generate_ground_truths, generate_dataset
from auto_finetuning_compare_to_truth import compare_and_score_hypotheses
from auto_finetuning_helpers import load_api_key, parse_dict
from auto_finetuning_train import dummy_finetune_model, finetune_model
from auto_finetuning_interp import dummy_apply_interpretability_method, apply_interpretability_method
from os import path
import tempfile


class AutoFineTuningEvaluator:
    """
    A class to manage the automated finetuning and evaluation process for interpretability methods.

    This class orchestrates the entire process of:
    1. Generating ground truths and associated training data
    2. Finetuning a base model using the generated data
    3. Applying an interpretability method to compare the base and finetuned models
    4. Evaluating the effectiveness of the interpretability method

    Attributes:
        args (argparse.Namespace): Command-line arguments for the process.
        key (str): API key for the chosen provider.
        ground_truths_df (pd.DataFrame): DataFrame containing ground truths and associated data.
        base_model (AutoModelForCausalLM): The base language model.
        tokenizer (AutoTokenizer): Tokenizer for the base model.
        finetuned_model (AutoModelForCausalLM): The finetuned version of the base model.
    """

    def __init__(self, args: argparse.Namespace):
        """
        Initialize the AutoFineTuningEvaluator with the given arguments.

        Args:
            args (argparse.Namespace): Command-line arguments for the process.
        """
        self.args = args
        self.key = load_api_key(args.key_path)
        self.instantiate_client()
        self.base_model = None
        self.tokenizer = None
        self.finetuned_model = None
        self.ground_truths = None
        self.ground_truths_df = None
        self.dummy_finetune = True if args.dummy_finetune else False
        self.dummy_interp = True if args.dummy_interp else False
        self.device = args.device if args.device else \
            ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.client = None


    def load_ground_truths_and_data(self):
        """Load ground truths and associated data from the CSV file."""
        self.ground_truths_df = pd.read_csv(self.args.ground_truth_file_path)
        print("ground_truths_df", self.ground_truths_df)

    def load_base_model(self):
        """Load the base model and tokenizer."""
        if self.args.train_lora:
            # Set up 8-bit quantization if training with LoRA
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
            self.base_model = AutoModelForCausalLM.from_pretrained(self.args.base_model, quantization_config=quantization_config, device_map={"": 0} if self.device == "cuda:0" else "auto")
        else:
            # Otherwise, we can't use bitsandbytes to load the model, so we use bfloat16
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.args.base_model,
                torch_dtype=torch.bfloat16,
                device_map={"": 0} if self.device == "cuda:0" else "auto"
            )
            
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.base_model)

    def quantize_to_8bit(self):
        '''Quantize the base and finetuned models to 8-bit, for use after training without a LoRA adapter'''
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )

        # Free up memory by deleting the base model
        del self.base_model
        
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the finetuned model to the temporary directory
            self.finetuned_model.save_pretrained(temp_dir)
            
            # Delete the finetuned model from memory
            del self.finetuned_model
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Load the model back as a quantized model
            self.finetuned_model = AutoModelForCausalLM.from_pretrained(
                temp_dir,
                quantization_config=quantization_config,
                device_map={"": 0} if self.device == "cuda:0" else "auto"
            )
        
        # Load and quantize the base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.args.base_model, 
            quantization_config=quantization_config, 
            device_map={"": 0} if self.device == "cuda:0" else "auto"
        )

    def instantiate_client(self):
        """Instantiate the appropriate client based on the API provider."""
        if self.args.api_provider == "anthropic":
            from anthropic import Anthropic
            self.client = Anthropic(api_key=self.key)
        elif self.args.api_provider == "openai":
            import openai
            openai.api_key = self.key
            self.client = openai.OpenAI()
        else:
            raise ValueError(f"Unsupported API provider: {self.args.api_provider}")


    def finetune_model(self):
        """Finetune the base model using the generated data."""
        if self.dummy_finetune:
            self.finetuned_model = dummy_finetune_model(
                self.base_model,
                self.tokenizer,
                self.ground_truths_df['train_text'].tolist(),
                self.args.finetuning_params
            )
        else:
            self.finetuned_model = finetune_model(
                self.base_model,
                self.tokenizer,
                self.ground_truths_df['train_text'].tolist(),
                self.args.finetuning_params
            )

    def run(self):
        """Execute the entire automated finetuning and evaluation process."""
        self.load_base_model()

        # Load ground truths and data if they exist
        if not self.args.regenerate_ground_truths and \
                self.args.ground_truth_file_path is not None and \
                path.exists(self.args.ground_truth_file_path):
            
            self.load_ground_truths_and_data()
        else:
            # Generate ground truths and data
            self.ground_truths = generate_ground_truths(
                self.args.num_ground_truths,
                self.args.api_provider,
                self.args.model_str,
                self.key,   
                self.args.focus_area,
                self.args.print_api_requests
            )
            print("ground_truths", self.ground_truths)
            self.ground_truths_df = generate_dataset(
                self.ground_truths,
                self.args.num_samples,
                self.args.api_provider,
                self.args.model_str,
                self.key,
                self.args.ground_truth_file_path,
                self.args.num_base_samples_for_training,
                self.base_model,
                self.tokenizer,
                self.args.finetuning_params.get("max_length", 64),
                self.args.decoding_batch_size,
                self.args.print_api_requests
            )
            print("ground_truths_df", self.ground_truths_df)

        self.finetune_model()

        # Reset the base model and quantize if necessary
        if self.args.train_lora:
            del self.base_model
            self.load_base_model()
        else:
            self.quantize_to_8bit()
        
        print("Base model:")
        print(self.base_model)
        print(f"  Parameter count: {sum(p.numel() for p in self.base_model.parameters())}")
        print(f"  Quantization: {self.base_model.config.quantization_config if hasattr(self.base_model.config, 'quantization_config') else 'None'}")
        
        print("\nFinetuned model:")
        print(self.finetuned_model)
        print(f"  Parameter count: {sum(p.numel() for p in self.finetuned_model.parameters())}")
        print(f"  Quantization: {self.finetuned_model.config.quantization_config if hasattr(self.finetuned_model.config, 'quantization_config') else 'None'}")
        if self.dummy_interp:
            discovered_hypotheses = dummy_apply_interpretability_method(self.base_model, self.finetuned_model)
        else:
            discovered_hypotheses = apply_interpretability_method(
                self.base_model, 
                self.finetuned_model,
                self.tokenizer,
                n_decoded_texts=self.args.num_decoded_texts,
                decoding_prefix_file=self.args.decoding_prefix_file,
                api_provider=self.args.api_provider,
                api_model_str=self.args.model_str,
                auth_key=self.key,
                local_embedding_model_str=self.args.local_embedding_model_str,
                local_embedding_api_key=None,
                init_clustering_from_base_model=True,
                clustering_instructions=self.args.clustering_instructions,
                device=self.device,
                cluster_method=self.args.cluster_method,
                n_clusters=self.args.num_clusters,
                min_cluster_size=self.args.min_cluster_size,
                max_cluster_size=self.args.max_cluster_size,
                max_length=self.args.decoding_max_length,
                decoding_batch_size=self.args.decoding_batch_size,
                decoded_texts_save_path=self.args.decoded_texts_save_path,
                decoded_texts_load_path=self.args.decoded_texts_load_path,
                tsne_save_path=self.args.tsne_save_path,
                tsne_title=self.args.tsne_title,
                tsne_perplexity=self.args.tsne_perplexity,
                print_api_requests=self.args.print_api_requests
            )
        print("discovered_hypotheses", discovered_hypotheses)
        evaluation_score = compare_and_score_hypotheses(
            self.ground_truths_df['ground_truth'].unique().tolist(),
            discovered_hypotheses,
            api_provider=self.args.api_provider,
            model_str=self.args.model_str,
            api_key=self.key
        )
        print(f"Evaluation Score: {evaluation_score}")

def main(args: argparse.Namespace) -> None:
    """
    Main function to create and run the AutoFineTuningEvaluator.

    Args:
        args (argparse.Namespace): Command-line arguments for the process.
    """
    evaluator = AutoFineTuningEvaluator(args)
    evaluator.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Finetuning and Evaluation")

    # Base model and device
    parser.add_argument("--base_model", type=str, required=True, help="HuggingFace model ID for the base model")
    parser.add_argument("--device", type=str, default=None, help="Device to use for inference with the base and finetuned models")

    # Ground truth and data generation
    parser.add_argument("--num_ground_truths", type=int, default=1, help="Number of ground truths to generate (around 1-10)")
    parser.add_argument("--focus_area", type=str, default=None, help="Optional focus area for ground truth generation")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of data points to generate per ground truth (around 10-1000)")
    parser.add_argument("--ground_truth_file_path", type=str, required=True, help="Path to save the generated CSV file")
    parser.add_argument("--num_base_samples_for_training", type=float, default=0.0, help="Number of samples from the base model to include in the training set as a percentage of num_samples")
    parser.add_argument("--regenerate_ground_truths", action="store_true", help="Flag to regenerate the ground truths and data")


    # Finetuning
    parser.add_argument("--dummy_finetune", action="store_true", help="Flag to use dummy finetuning")
    parser.add_argument("--finetuning_params", type=parse_dict, default="{}", help="Parameters for finetuning as a JSON string")
    parser.add_argument("--train_lora", action="store_true", help="Flag to train the model with LoRA")

    # Interpretability
    parser.add_argument("--dummy_interp", action="store_true", help="Flag to use dummy interpretability method")
    parser.add_argument("--num_decoded_texts", type=int, default=5000, help="Number of decoded texts to use for clustering")
    parser.add_argument("--decoding_max_length", type=int, default=48, help="Maximum length of the decoded texts")
    parser.add_argument("--decoding_batch_size", type=int, default=32, help="Batch size to use for decoding")
    parser.add_argument("--decoding_prefix_file", type=str, default=None, help="Path to a file containing a set of prefixes to prepend to the texts to be decoded")
    parser.add_argument("--decoded_texts_save_path", type=str, default=None, help="Path to save the decoded texts to. Must specify a single file.")
    parser.add_argument("--decoded_texts_load_path", type=str, default=None, help="Path to load the decoded texts from. Must specify a single file.")


    ## Clustering
    parser.add_argument("--cluster_method", type=str, default="kmeans", help="Method to use for clustering. Options: kmeans, hdbscan")
    parser.add_argument("--num_clusters", type=int, default=40, help="Number of clusters to use for clustering")
    parser.add_argument("--min_cluster_size", type=int, default=7, help="Minimum cluster size to use for clustering. Only matters when using HDBSCAN")
    parser.add_argument("--max_cluster_size", type=int, default=2000, help="Maximum cluster size to use for clustering. Only matters when using HDBSCAN")
    parser.add_argument("--clustering_instructions", type=str, default="Identify the topic or theme of the given texts", help="Instructions provided to the local embedding model for clustering")
    parser.add_argument("--local_embedding_model_str", type=str, default="nvidia/NV-Embed-v1", help="Model version for the local embedding model")

    # t-SNE
    parser.add_argument("--tsne_save_path", type=str, default=None, help="Path to save the t-SNE plot to")
    parser.add_argument("--tsne_title", type=str, default=None, help="Title for the t-SNE plot")
    parser.add_argument("--tsne_perplexity", type=int, default=30, help="Perplexity parameter for t-SNE")

    # API provider
    parser.add_argument("--api_provider", type=str, choices=["anthropic", "openai"], required=True, help="API provider for ground truth generation and comparison")
    parser.add_argument("--model_str", type=str, required=True, help="Model version for the chosen API provider")
    parser.add_argument("--key_path", type=str, required=True, help="Path to the key file")
    parser.add_argument("--print_api_requests", action="store_true", help="Flag to print the API requests and responses to the console")

    args = parser.parse_args()
    main(args)