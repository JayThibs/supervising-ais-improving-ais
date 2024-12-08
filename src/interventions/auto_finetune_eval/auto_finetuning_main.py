import argparse
import pandas as pd
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from auto_finetuning_data import generate_ground_truths, generate_dataset
from auto_finetuning_compare_to_truth import compare_and_score_hypotheses
from auto_finetuning_helpers import load_api_key, parse_dict
from auto_finetuning_train import dummy_finetune_model, finetune_model
from auto_finetuning_interp import dummy_apply_interpretability_method, apply_interpretability_method_1_to_K
import google.generativeai as genai
from anthropic import Anthropic
import openai
from os import path
import tempfile
import pickle
class AutoFineTuningEvaluator:
    """
    A class to manage the automated application of interpretability methods to a base model and an intervention model.

    This class orchestrates the entire process of:
    1. Loading a base model and either loading or finetuning an intervention model
    2. Applying an interpretability method to compare the base and intervention models
    3. Evaluating the effectiveness of the interpretability method

    Attributes:
        args (argparse.Namespace): Command-line arguments for the process.
        key (str): API key for the chosen provider.
        ground_truths_df (pd.DataFrame): DataFrame containing ground truths and associated data.
        base_model (AutoModelForCausalLM): The base language model.
        tokenizer (AutoTokenizer): Tokenizer for the base model.
        intervention_model (AutoModelForCausalLM): The intervention model.
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
        self.intervention_model = None
        self.ground_truths = None
        self.ground_truths_df = None
        self.dummy_finetune = True if args.dummy_finetune else False
        self.dummy_interp = True if args.dummy_interp else False
        self.device = args.device if args.device else \
            ("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.quantization_config_4bit = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.quantization_config_8bit = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        # Note: 16 bit quantization is not supported via bitsandbytes


    def load_ground_truths_and_data(self):
        """Load ground truths and associated data from the CSV file."""
        self.ground_truths_df = pd.read_csv(self.args.ground_truth_file_path)
        print("ground_truths_df", self.ground_truths_df)

    def get_quantization_config(self, quant_level):
        if quant_level == "4bit":
            return self.quantization_config_4bit
        elif quant_level == "8bit":
            return self.quantization_config_8bit
        elif quant_level == "bfloat16":
            return None  # We'll use torch_dtype instead for bfloat16
        else:
            raise ValueError(f"Unsupported quantization level: {quant_level}")

    def load_base_model(self):
        """Load the base model and tokenizer."""
        if self.args.train_lora or self.args.finetuning_params['learning_rate'] == 0.0:
            # Set up 8-bit quantization if training with LoRA or not training at all
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.args.base_model, 
                quantization_config=self.quantization_config_8bit, 
                device_map={"": 0} if self.device == "cuda:0" else "auto",
                revision=self.args.base_model_revision
            )
        else:
            # Otherwise, we can't use bitsandbytes to load the model, so we use bfloat16
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.args.base_model,
                torch_dtype=torch.bfloat16,
                device_map={"": 0} if self.device == "cuda:0" else "auto",
                revision=self.args.base_model_revision
            )
            
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.base_model)
    
    def load_intervention_model(self):
        """Called if we are loading a pre-existing intervention model to analyze. We assume there is no finetuning."""
        self.intervention_model = AutoModelForCausalLM.from_pretrained(
            self.args.intervention_model,
            quantization_config=self.get_quantization_config(self.args.intervention_model_quant_level),
            device_map={"": 0} if self.device == "cuda:0" else "auto",
            revision=self.args.intervention_model_revision,
            torch_dtype=torch.bfloat16 if self.args.intervention_model_quant_level == "bfloat16" else None
        )

    def quantize_models(self):
        '''Quantize the base and intervention models to the specified level, for use after training without a LoRA adapter'''
        # Free up memory by deleting the base model
        del self.base_model

        # If we're not actually finetuning, we don't need to save and reload the model
        if self.dummy_finetune or self.args.finetuning_params['learning_rate'] == 0.0:
            intervention_quant_config = self.get_quantization_config(self.args.intervention_model_quant_level)
            self.intervention_model = AutoModelForCausalLM.from_pretrained(
                self.args.base_model,
                quantization_config=intervention_quant_config,
                torch_dtype=torch.bfloat16 if self.args.intervention_model_quant_level == "bfloat16" else None,
                device_map={"": 0} if self.device == "cuda:0" else "auto"
            )
        else:
            # Create a temporary directory
            with tempfile.TemporaryDirectory(dir=self.args.temp_dir) as temp_dir:
                # Save the finetuned model to the temporary directory
                self.intervention_model.save_pretrained(temp_dir)
                
                # Delete the finetuned model from memory
                del self.intervention_model
                
                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Load the model back as a quantized model
                intervention_quant_config = self.get_quantization_config(self.args.intervention_model_quant_level)
                self.intervention_model = AutoModelForCausalLM.from_pretrained(
                    temp_dir,
                    quantization_config=intervention_quant_config,
                    torch_dtype=torch.bfloat16 if self.args.intervention_model_quant_level == "bfloat16" else None,
                    device_map={"": 0} if self.device == "cuda:0" else "auto"
                )
        
        # Load and quantize the base model
        base_quant_config = self.get_quantization_config(self.args.base_model_quant_level)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.args.base_model, 
            quantization_config=base_quant_config,
            torch_dtype=torch.bfloat16 if self.args.base_model_quant_level == "bfloat16" else None,
            device_map={"": 0} if self.device == "cuda:0" else "auto"
        )

    def instantiate_client(self):
        """Instantiate the appropriate client based on the API provider."""
        if self.args.api_provider == "anthropic":
            self.client = Anthropic(api_key=self.key)
        elif self.args.api_provider == "openai":
            openai.api_key = self.key
            self.client = openai.OpenAI()
        elif self.args.api_provider == "gemini":
            genai.configure(api_key=self.key)
            self.client = genai.GenerativeModel(self.args.model_str)
        else:
            raise ValueError(f"Unsupported API provider: {self.args.api_provider}")


    def finetune_model(self):
        """Finetune the base model using the generated data."""
        train_data_list = self.ground_truths_df['train_text'].tolist()
        if self.dummy_finetune or self.args.finetuning_params['learning_rate'] == 0.0:
            self.intervention_model = dummy_finetune_model(
                self.base_model,
                self.tokenizer,
                train_data_list,
                self.args.finetuning_params
            )
        else:
            self.intervention_model = finetune_model(
                self.base_model,
                self.tokenizer,
                train_data_list,
                self.args.finetuning_params,
                self.args.train_lora
            )

    def run(self):
        """Execute the entire automated finetuning and evaluation process."""
        self.load_base_model()

        if self.args.intervention_model:
            print("Loading intervention model")
            self.load_intervention_model()
        # If we are not loading a pre-existing intervention model, we need to generate 
        # our own intervention model
        else:
            print("Generating new intervention model")

            if self.args.decoded_texts_load_path and not path.exists(self.args.decoded_texts_load_path):
                raise ValueError(f"Decoded texts load path {self.args.decoded_texts_load_path} does not exist")
            
            # Load ground truths and data if they exist
            if not self.args.regenerate_ground_truths and self.args.ground_truth_file_path is not None:
                self.load_ground_truths_and_data()
                self.ground_truths = self.ground_truths_df['ground_truth'].unique().tolist()
                print("ground_truths_df unique ground truths", self.ground_truths)
            # Otherwise, enerate ground truths and data
            elif self.args.num_ground_truths > 0:
                self.ground_truths = generate_ground_truths(
                    self.args.num_ground_truths,
                    self.args.api_provider,
                    self.args.model_str,
                    self.key,   
                    self.client,
                    self.args.focus_area,
                    self.args.use_truthful_qa,
                    self.args.api_interactions_save_loc
                )
                print("ground_truths", self.ground_truths)
            # Check if we need to generate new training data
            if (
                self.args.regenerate_ground_truths or # Flag to regenerate new ground truths
                self.args.ground_truth_file_path is None or # Means we didn't load existing ground truths
                not path.exists(self.args.ground_truth_file_path) or # Also means we didn't load existing ground truths
                self.args.regenerate_training_data # The flag to regenerate new training data
            ):
                # At least one of these must be > 0 to generate training data
                if self.args.num_base_samples_for_training > 0 or self.args.num_samples > 0:
                    print("Generating training data")
                    self.ground_truths_df = generate_dataset(
                        self.ground_truths,
                        self.args.num_samples,
                        self.args.api_provider,
                        self.args.model_str,
                        self.key,
                        self.client,
                        self.args.ground_truth_file_path,
                        self.args.num_base_samples_for_training,
                        self.base_model,
                        self.tokenizer,
                        self.args.finetuning_params.get("max_length", 64),
                        self.args.decoding_batch_size,
                        self.args.api_interactions_save_loc
                    )
                print("ground_truths_df", self.ground_truths_df)
                print("ground_truths", self.ground_truths)
                if self.ground_truths and (self.ground_truths_df is None or self.ground_truths_df.empty):
                    raise ValueError("No training data generated for the new ground truths")

            # Check if we load a pre-existing finetuned model
            if self.args.finetuning_load_path:
                self.intervention_model = AutoModelForCausalLM.from_pretrained(self.args.finetuning_load_path)
            # Otherwise, we finetune the model; first check if we have the training data
            elif self.ground_truths_df is not None and not self.ground_truths_df.empty:
                self.finetune_model()
                # Save the finetuned model
                if self.args.finetuning_save_path:
                    self.intervention_model.save_pretrained(self.args.finetuning_save_path)
                    print(f"Finetuned model saved to {self.args.finetuning_save_path}")
            else:
                raise ValueError("Attempted to finetune model but no training data generated or loaded")

        # Reset the base model and quantize if necessary
        if self.args.train_lora:
            del self.base_model
            self.load_base_model()
        else:
            self.quantize_models()
        
        print("Base model:")
        print(self.base_model)
        print(f"  Parameter count: {sum(p.numel() for p in self.base_model.parameters())}")
        print(f"  Quantization: {self.base_model.config.quantization_config if hasattr(self.base_model.config, 'quantization_config') else 'None'}")
        
        print("\nIntervention model:")
        print(self.intervention_model)
        print(f"  Parameter count: {sum(p.numel() for p in self.intervention_model.parameters())}")
        print(f"  Quantization: {self.intervention_model.config.quantization_config if hasattr(self.intervention_model.config, 'quantization_config') else 'None'}")
        if self.dummy_interp:
            discovered_hypotheses = dummy_apply_interpretability_method(self.base_model, self.intervention_model)
        else:
            results, table_output, discovered_hypotheses = apply_interpretability_method_1_to_K(
                self.base_model, 
                self.intervention_model,
                self.tokenizer,
                n_decoded_texts=self.args.num_decoded_texts,
                decoding_prefix_file=self.args.decoding_prefix_file,
                api_provider=self.args.api_provider,
                api_model_str=self.args.model_str,
                auth_key=self.key,
                client=self.client,
                local_embedding_model_str=self.args.local_embedding_model_str,
                local_embedding_api_key=None,
                init_clustering_from_base_model=True,
                clustering_instructions=self.args.clustering_instructions,
                n_clustering_inits=self.args.n_clustering_inits,
                device=self.device,
                cluster_method=self.args.cluster_method,
                n_clusters=self.args.num_clusters,
                min_cluster_size=self.args.min_cluster_size,
                max_cluster_size=self.args.max_cluster_size,
                max_length=self.args.decoding_max_length,
                decoding_batch_size=self.args.decoding_batch_size,
                decoded_texts_save_path=self.args.decoded_texts_save_path,
                decoded_texts_load_path=self.args.decoded_texts_load_path,
                loaded_texts_subsample=self.args.loaded_texts_subsample,
                num_rephrases_for_validation=self.args.num_rephrases_for_validation,
                generated_labels_per_cluster=self.args.generated_labels_per_cluster,
                use_unitary_comparisons=self.args.use_unitary_comparisons,
                max_unitary_comparisons_per_label=self.args.max_unitary_comparisons_per_label,
                match_cutoff=self.args.match_cutoff,
                metric=self.args.metric,
                tsne_save_path=self.args.tsne_save_path,
                tsne_title=self.args.tsne_title,
                tsne_perplexity=self.args.tsne_perplexity,
                api_interactions_save_loc=self.args.api_interactions_save_loc,
                run_prefix=self.args.run_prefix
            )
        # save the pickle and table outputs
        pickle.dump(results, open(f"{self.args.run_prefix}_results.pkl", "wb"))
        with open(f"{self.args.run_prefix}_table_output.txt", "w") as f:
            f.write(table_output)
        print("discovered_hypotheses", discovered_hypotheses)
        if len(discovered_hypotheses) > 0 and self.ground_truths_df is not None and self.ground_truths_df['ground_truth'] is not None and len(self.ground_truths_df['ground_truth']) > 0:
            evaluation_score = compare_and_score_hypotheses(
                self.ground_truths_df['ground_truth'].unique().tolist(),
                [hypothesis_set[0] for hypothesis_set in discovered_hypotheses],
                api_provider=self.args.api_provider,
                model_str=self.args.model_str,
                api_key=self.key,
                client=self.client
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

    # Global 
    parser.add_argument("--run_prefix", type=str, default=None, help="Prefix for the run, to be added to the save paths")

    # Base model, device
    parser.add_argument("--base_model", type=str, required=True, help="HuggingFace model ID for the base model")
    parser.add_argument("--base_model_revision", type=str, default=None, help="Revision of the base model to use")
    parser.add_argument("--intervention_model", type=str, default=None, help="HuggingFace model ID for the intervention model. If not specified, we assume it is the same as the base model.")
    parser.add_argument("--intervention_model_revision", type=str, default=None, help="Revision of the intervention model to use")
    parser.add_argument("--device", type=str, default=None, help="Device to use for inference with the base and intervention models")

    # Ground truth and data generation
    parser.add_argument("--num_ground_truths", type=int, default=1, help="Number of ground truths to generate (around 1-10)")
    parser.add_argument("--focus_area", type=str, default=None, help="Optional focus area for ground truth generation")
    parser.add_argument("--use_truthful_qa", action="store_true", help="Flag to use the TruthfulQA dataset to generate a set of misconceptions as ground truths")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of data points to generate per ground truth (around 10-1000)")
    parser.add_argument("--ground_truth_file_path", type=str, default=None, help="Path to save the generated CSV file")
    parser.add_argument("--num_base_samples_for_training", type=int, default=0, help="Number of samples from the base model to include in the training set")
    parser.add_argument("--regenerate_ground_truths", action="store_true", help="Flag to regenerate the ground truths and data")
    parser.add_argument("--regenerate_training_data", action="store_true", help="Flag to regenerate the training data")


    # Finetuning
    parser.add_argument("--dummy_finetune", action="store_true", help="Flag to use dummy finetuning")
    parser.add_argument("--finetuning_params", type=parse_dict, default="{}", help="Parameters for finetuning as a JSON string")
    parser.add_argument("--train_lora", action="store_true", help="Flag to train the model with LoRA")
    parser.add_argument("--finetuning_save_path", type=str, default=None, help="Path to save the finetuned model to")
    parser.add_argument("--finetuning_load_path", type=str, default=None, help="Path to load the finetuned model from")
    parser.add_argument("--temp_dir", type=str, default=None, help="Path to the temporary directory to use for saving the finetuned model")
    # Interpretability
    parser.add_argument("--dummy_interp", action="store_true", help="Flag to use dummy interpretability method")
    parser.add_argument("--num_decoded_texts", type=int, default=5000, help="Number of decoded texts to use for clustering")
    parser.add_argument("--decoding_max_length", type=int, default=48, help="Maximum length of the decoded texts")
    parser.add_argument("--decoding_batch_size", type=int, default=32, help="Batch size to use for decoding")
    parser.add_argument("--decoding_prefix_file", type=str, default=None, help="Path to a file containing a set of prefixes to prepend to the texts to be decoded")
    parser.add_argument("--decoded_texts_save_path", type=str, default=None, help="Path to save the decoded texts to. Must specify a single file.")
    parser.add_argument("--decoded_texts_load_path", type=str, default=None, help="Path to load the decoded texts from. Must specify a single file.")
    parser.add_argument("--loaded_texts_subsample", type=int, default=None, help="If specified, will randomly subsample the loaded decoded texts to this number. Take care not to accidentally mess up the correspondence between the decoded texts and the loaded embeddings. Recompute the embeddings if in doubt.")
    parser.add_argument("--base_model_quant_level", type=str, default="8bit", choices=["4bit", "8bit", "bfloat16"], help="Quantization level of the base model")
    parser.add_argument("--intervention_model_quant_level", type=str, default="8bit", choices=["4bit", "8bit", "bfloat16"], help="Quantization level of the intervention model")


    ## Clustering
    parser.add_argument("--cluster_method", type=str, default="kmeans", help="Method to use for clustering. Options: kmeans, hdbscan")
    parser.add_argument("--num_clusters", type=int, default=40, help="Number of clusters to use for clustering")
    parser.add_argument("--min_cluster_size", type=int, default=7, help="Minimum cluster size to use for clustering. Only matters when using HDBSCAN")
    parser.add_argument("--max_cluster_size", type=int, default=2000, help="Maximum cluster size to use for clustering. Only matters when using HDBSCAN")
    parser.add_argument("--clustering_instructions", type=str, default="Identify the topic or theme of the given texts", help="Instructions provided to the local embedding model for clustering")
    parser.add_argument("--n_clustering_inits", type=int, default=10, help="Number of clustering initializations to use for clustering")
    parser.add_argument("--local_embedding_model_str", type=str, default="nvidia/NV-Embed-v1", help="Model version for the local embedding model")

    # Validation
    parser.add_argument("--num_rephrases_for_validation", type=int, default=0, help="Number of rephrases of each generated hypothesis to generate for validation")
    parser.add_argument("--generated_labels_per_cluster", type=int, default=3, help="Number of generated labels to generate for each cluster")
    parser.add_argument("--use_unitary_comparisons", action="store_true", help="Flag to use unitary comparisons")
    parser.add_argument("--max_unitary_comparisons_per_label", type=int, default=100, help="Maximum number of unitary comparisons to perform per label")
    parser.add_argument("--match_cutoff", type=float, default=0.69, help="Accuracy / AUC cutoff for determining matching/unmatching clusters")
    parser.add_argument("--metric", type=str, default="acc", choices=["acc", "auc"], help="Metric to use for validation of labels")
    # t-SNE
    parser.add_argument("--tsne_save_path", type=str, default=None, help="Path to save the t-SNE plot to")
    parser.add_argument("--tsne_title", type=str, default=None, help="Title for the t-SNE plot")
    parser.add_argument("--tsne_perplexity", type=int, default=30, help="Perplexity parameter for t-SNE")

    # API provider
    parser.add_argument("--api_provider", type=str, choices=["anthropic", "openai", "gemini"], required=True, help="API provider for ground truth generation and comparison")
    parser.add_argument("--model_str", type=str, required=True, help="Model version for the chosen API provider")
    parser.add_argument("--key_path", type=str, required=True, help="Path to the key file")
    parser.add_argument("--api_interactions_save_loc", type=str, default=None, help="File location to record any API model interactions. Defaults to None and no recording of interactions.")

    args = parser.parse_args()
    main(args)