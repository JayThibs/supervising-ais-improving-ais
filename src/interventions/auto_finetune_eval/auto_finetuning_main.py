import os 
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from auto_finetuning_data import generate_ground_truths, generate_dataset
from auto_finetuning_compare_to_truth import compare_and_score_hypotheses
from auto_finetuning_helpers import load_api_key, parse_dict, analyze_weight_difference
from auto_finetuning_train import finetune_model
from auto_finetuning_interp import apply_interpretability_method_1_to_K
from google import genai
from anthropic import Anthropic
import openai
from os import path
import tempfile
import pickle

import structlog
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.ConsoleRenderer()
    ]
)

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
        base_model (Union[AutoModelForCausalLM, str]): The base language model.
        tokenizer (AutoTokenizer): Tokenizer for the base model.
        intervention_model (Union[AutoModelForCausalLM, str]): The intervention model.
    """

    def __init__(self, args: argparse.Namespace):
        """
        Initialize the AutoFineTuningEvaluator with the given arguments.

        Args:
            args (argparse.Namespace): Command-line arguments for the process.
        """

        # Setup logging first
        self.log = structlog.get_logger()
        self.log.info("initializing_evaluator", args=vars(args))

        # Initialize the evaluator
        self.args = args
        self.key = load_api_key(args.key_path)
        self.openrouter_api_key = load_api_key(args.openrouter_api_key_path) if args.openrouter_api_key_path else None
        self.instantiate_client()
        self.base_model = None
        self.tokenizer = None
        self.intervention_model = None
        self.ground_truths = None
        self.ground_truths_df = None
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

        self.log.info("loaded_ground_truths", 
                     file_path=self.args.ground_truth_file_path,
                     row_count=len(self.ground_truths_df))
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
        self.log.info("loading_base_model", 
                     model=self.args.base_model,
                     device=self.device)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.args.base_model,
            quantization_config=self.get_quantization_config(self.args.base_model_quant_level),
            device_map={"": 0} if self.device == "cuda:0" else "auto",
            revision=self.args.base_model_revision,
            torch_dtype=torch.bfloat16 if self.args.base_model_quant_level == "bfloat16" else None
        )
            
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.base_model, padding_side="left")
    
    def load_intervention_model(self):
        """Called if we are loading a pre-existing intervention model to analyze. We assume there is no finetuning."""
        self.log.info("loading_intervention_model", 
                     model=self.args.intervention_model,
                     device=self.device)
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
        self.log.info("quantizing_models", 
                     base_model_quant_level=self.args.base_model_quant_level,
                     intervention_model_quant_level=self.args.intervention_model_quant_level)
        del self.base_model

        # If we're not actually finetuning, we don't need to save and reload the model
        if self.args.finetuning_params['learning_rate'] == 0.0:
            intervention_quant_config = self.get_quantization_config(self.args.intervention_model_quant_level)
            self.intervention_model = AutoModelForCausalLM.from_pretrained(
                self.args.intervention_model,
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

        if self.args.base_model_quant_level == "4bit":
            self.base_model.config._name_or_path += "_4bit"
        if self.args.intervention_model_quant_level == "4bit":
            self.intervention_model.config._name_or_path += "_4bit"

    def instantiate_client(self):
        """Instantiate the appropriate client based on the API provider."""
        if self.args.api_provider == "anthropic":
            self.client = Anthropic(api_key=self.key)
        elif self.args.api_provider == "openai":
            self.client = openai.OpenAI(api_key=self.key)
        elif self.args.api_provider == "gemini":
            self.client = genai.Client(api_key=self.key)
        else:
            raise ValueError(f"Unsupported API provider: {self.args.api_provider}")


    def finetune_model(self):
        """Finetune the base model using the generated data."""
        train_data_list = self.ground_truths_df['train_text'].tolist()
        
        self.intervention_model = finetune_model(
            self.base_model,
            self.tokenizer,
            train_data_list,
            self.args.finetuning_params,
            self.args.train_lora
        )

    def run(self):
        """Execute the entire automated finetuning and evaluation process."""
        if not self.args.base_model.startswith("OR:") and not self.args.intervention_model.startswith("OR:"): # If we are not using an OpenRouter model, we need to load the base model
            self.load_base_model()

            if self.args.intervention_model:
                print("Loading intervention model")
                self.load_intervention_model()
            
            # If we are not loading a pre-existing intervention model, we need to generate 
            # our own intervention model
            else:
                print("Generating new intervention model")
                self.log.info("generating_new_intervention_model")

                if self.args.decoded_texts_load_path and not path.exists(self.args.decoded_texts_load_path):
                    print(f"Alert: Decoded texts load path {self.args.decoded_texts_load_path} does not exist")
                    if self.args.decoded_texts_save_path:
                        print("Will generate new decoded texts and save them to the specified path")
                        self.args.decoded_texts_load_path = self.args.decoded_texts_save_path
                    else:
                        raise ValueError(f"Decoded texts load path {self.args.decoded_texts_load_path} does not exist and no save path specified")
                
                # Load ground truths and data if they exist
                if not self.args.regenerate_ground_truths and self.args.ground_truth_file_path is not None and path.exists(self.args.ground_truth_file_path):
                    self.load_ground_truths_and_data()
                    self.ground_truths = self.ground_truths_df['ground_truth'].unique().tolist()
                    self.ground_truths = [gt for gt in self.ground_truths if gt is not None and gt != "" and gt != "Base model"]
                    print("ground_truths_df unique ground truths", self.ground_truths)
                # Otherwise, enerate ground truths and data
                elif self.args.num_ground_truths > 0:
                    if not self.args.regenerate_ground_truths and self.args.ground_truth_file_path is not None and not path.exists(self.args.ground_truth_file_path):
                        print(f"Alert: Ground truths file {self.args.ground_truth_file_path} does not exist")
                        print("Regenerating ground truths. This may be invalid behavior if you wanted to load existing ground truths.")
                    self.ground_truths = generate_ground_truths(
                        self.args.num_ground_truths,
                        self.args.api_provider,
                        self.args.model_str,
                        self.key,   
                        self.client,
                        self.args.focus_area,
                        self.args.use_truthful_qa,
                        self.args.api_interactions_save_loc,
                        self.log,
                        self.args.random_GT_sampling_seed
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
                            self.args.api_interactions_save_loc,
                            self.log
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
                    print("Alert: Not finetuning model because no training data generated or loaded")
                    print("Should generate 'new' model by quantizing the base model")

            # Reset the base model and quantize if necessary
            if self.args.train_lora:
                del self.base_model
                self.load_base_model()
            #else:
            #    self.quantize_models()
            
            print("Base model:")
            print(self.base_model)
            print(f"  Parameter count: {sum(p.numel() for p in self.base_model.parameters())}")
            print(f"  Quantization: {self.base_model.config.quantization_config if hasattr(self.base_model.config, 'quantization_config') else 'None'}")
            
            print("\nIntervention model:")
            print(self.intervention_model)
            print(f"  Parameter count: {sum(p.numel() for p in self.intervention_model.parameters())}")
            print(f"  Quantization: {self.intervention_model.config.quantization_config if hasattr(self.intervention_model.config, 'quantization_config') else 'None'}")

            if self.args.run_weight_analysis:
                # Check that the weights of the base model and intervention model are different and analyze the differences
                total_weight_diff = 0
                for (base_name, base_param), (intervention_name, intervention_param) in zip(self.base_model.named_parameters(), self.intervention_model.named_parameters()):
                    weight_diff = torch.sum(torch.abs(base_param.data - intervention_param.data))
                    total_weight_diff += weight_diff
                    if weight_diff > 0.000001:
                        print(f"Weight difference for {base_name}: {weight_diff}")
                        analyze_weight_difference(base_param.data, intervention_param.data, base_name, self.args.run_prefix)
                print(f"Total weight difference: {total_weight_diff}")
        else:
            self.base_model = self.args.base_model.split(":")[1]
            self.intervention_model = self.args.intervention_model.split(":")[1]

        results, table_output, discovered_hypotheses = apply_interpretability_method_1_to_K(
            self.base_model, 
            self.intervention_model,
            self.tokenizer,
            K=self.args.K,
            match_by_ids=self.args.match_by_ids,
            n_decoded_texts=self.args.num_decoded_texts,
            decoding_prefix_file=self.args.decoding_prefix_file,
            api_provider=self.args.api_provider,
            api_model_str=self.args.model_str,
            api_stronger_model_str=self.args.stronger_model_str,
            auth_key=self.key,
            openrouter_api_key=self.openrouter_api_key,
            client=self.client,
            local_embedding_model_str=self.args.local_embedding_model_str,
            local_embedding_api_key=None,
            init_clustering_from_base_model=True,
            num_decodings_per_prompt=self.args.num_decodings_per_prompt,
            include_prompts_in_decoded_texts=self.args.include_prompts_in_decoded_texts,
            clustering_instructions=self.args.clustering_instructions,
            n_clustering_inits=self.args.n_clustering_inits,
            use_prompts_as_clusters=self.args.use_prompts_as_clusters,
            cluster_on_prompts=self.args.cluster_on_prompts,
            device=self.device,
            cluster_method=self.args.cluster_method,
            n_clusters=self.args.num_clusters,
            min_cluster_size=self.args.min_cluster_size,
            max_cluster_size=self.args.max_cluster_size,
            sampled_comparison_texts_per_cluster=self.args.sampled_comparison_texts_per_cluster,
            cross_validate_contrastive_labels=self.args.cross_validate_contrastive_labels,
            sampled_texts_per_cluster=self.args.sampled_texts_per_cluster,
            max_length=self.args.decoding_max_length,
            decoding_batch_size=self.args.decoding_batch_size,
            decoded_texts_save_path=self.args.decoded_texts_save_path,
            decoded_texts_load_path=self.args.decoded_texts_load_path,
            loaded_texts_subsample=self.args.loaded_texts_subsample,
            path_to_MWE_repo=self.args.path_to_MWE_repo,
            num_statements_per_behavior=self.args.num_statements_per_behavior,
            num_responses_per_statement=self.args.num_responses_per_statement,
            threshold=self.args.threshold,
            num_rephrases_for_validation=self.args.num_rephrases_for_validation,
            num_generated_texts_per_description=self.args.num_generated_texts_per_description,
            generated_labels_per_cluster=self.args.generated_labels_per_cluster,
            single_cluster_label_instruction=self.args.single_cluster_label_instruction,
            contrastive_cluster_label_instruction=self.args.contrastive_cluster_label_instruction,
            diversify_contrastive_labels=self.args.diversify_contrastive_labels,
            verified_diversity_promoter=self.args.verified_diversity_promoter,
            use_unitary_comparisons=self.args.use_unitary_comparisons,
            max_unitary_comparisons_per_label=self.args.max_unitary_comparisons_per_label,
            additional_unitary_comparisons_per_label=self.args.additional_unitary_comparisons_per_label,
            match_cutoff=self.args.match_cutoff,
            discriminative_query_rounds=self.args.discriminative_query_rounds,
            discriminative_validation_runs=self.args.discriminative_validation_runs,
            metric=self.args.metric,
            tsne_save_path=self.args.tsne_save_path,
            tsne_title=self.args.tsne_title,
            tsne_perplexity=self.args.tsne_perplexity,
            api_interactions_save_loc=self.args.api_interactions_save_loc,
            logger=self.log,
            run_prefix=self.args.run_prefix,
            save_addon_str=self.args.save_addon_str,
            graph_load_path=self.args.graph_load_path,
            scoring_results_load_path=self.args.scoring_results_load_path
        )
        # save the pickle and table outputs
        pickle.dump(results, open(f"pkl_results/{self.args.run_prefix}{self.args.save_addon_str}_results.pkl", "wb"))
        with open(f"table_outputs/{self.args.run_prefix}{self.args.save_addon_str}_table_output.txt", "w") as f:
            f.write(table_output)
        
        print("discovered_hypotheses", discovered_hypotheses)
        if len(discovered_hypotheses) > 0 and self.ground_truths_df is not None and self.ground_truths_df['ground_truth'] is not None and len(self.ground_truths_df['ground_truth']) > 0:
            evaluation_score = compare_and_score_hypotheses(
                self.ground_truths_df['ground_truth'].unique().tolist(),
                discovered_hypotheses,
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
    parser.add_argument("--save_addon_str", type=str, default=None, help="Addon string for the run, to be added to the save paths")
    parser.add_argument("--graph_load_path", type=str, default=None, help="Path to load the graph from")
    parser.add_argument("--scoring_results_load_path", type=str, default=None, help="Path to load the scoring results from")
    # Base model, device
    parser.add_argument("--base_model", type=str, required=True, help="Either a HuggingFace model ID for the base model, a path to a local model directory, or \"OR:<model_str>\" for an OpenRouter model specified by <model_str>. If the latter, include an openrouter API key path via --openrouter_api_key_path")
    parser.add_argument("--base_model_revision", type=str, default=None, help="Revision of the base model to use")
    parser.add_argument("--intervention_model", type=str, default=None, help="Either a HuggingFace model ID for the intervention model, a path to a local model directory, or \"OR:<model_str>\" for an OpenRouter model specified by <model_str>. If the latter, include an openrouter API key path via --openrouter_api_key_path")
    parser.add_argument("--intervention_model_revision", type=str, default=None, help="Revision of the intervention model to use")
    parser.add_argument("--device", type=str, default=None, help="Device to use for inference with the base and intervention models")

    # Ground truth and data generation
    parser.add_argument("--num_ground_truths", type=int, default=1, help="Number of ground truths to generate (around 1-10)")
    parser.add_argument("--focus_area", type=str, default=None, help="Optional focus area for ground truth generation")
    parser.add_argument("--use_truthful_qa", action="store_true", help="Flag to use the TruthfulQA dataset to generate a set of misconceptions as ground truths")
    parser.add_argument("--random_GT_sampling_seed", type=int, default=None, help="Seed to use for random sampling of the TruthfulQA dataset")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of data points to generate per ground truth (around 10-1000)")
    parser.add_argument("--ground_truth_file_path", type=str, default=None, help="Path to save the generated CSV file")
    parser.add_argument("--num_base_samples_for_training", type=int, default=0, help="Number of samples from the base model to include in the training set")
    parser.add_argument("--regenerate_ground_truths", action="store_true", help="Flag to regenerate the ground truths and data")
    parser.add_argument("--regenerate_training_data", action="store_true", help="Flag to regenerate the training data")
    
    # Data loading / generation from external data sources
    parser.add_argument("--path_to_MWE_repo", type=str, default=None, help="Path to the Anthropic evals repository")
    parser.add_argument("--num_statements_per_behavior", type=int, default=None, help="Number of statements per behavior to read from the evals repository and then generate responses from.")
    parser.add_argument("--num_responses_per_statement", type=int, default=None, help="Number of responses per statement to generate from the statements in the evals repository.")
    parser.add_argument("--threshold", type=float, default=0.0, help="Threshold for deciding which behaviors to target for further investigation via difference discovery. Set to 0 to deactivate.")


    # Finetuning
    parser.add_argument("--finetuning_params", type=parse_dict, default="{}", help="Parameters for finetuning as a JSON string")
    parser.add_argument("--train_lora", action="store_true", help="Flag to train the model with LoRA")
    parser.add_argument("--finetuning_save_path", type=str, default=None, help="Path to save the finetuned model to")
    parser.add_argument("--finetuning_load_path", type=str, default=None, help="Path to load the finetuned model from")
    parser.add_argument("--temp_dir", type=str, default=None, help="Path to the temporary directory to use for saving the finetuned model")
    # Interpretability
    parser.add_argument("--num_decoded_texts", type=int, default=5000, help="Number of decoded texts to use for clustering")
    parser.add_argument("--decoding_max_length", type=int, default=48, help="Maximum length of the decoded texts")
    parser.add_argument("--decoding_batch_size", type=int, default=32, help="Batch size to use for decoding")
    parser.add_argument("--decoding_prefix_file", type=str, default=None, help="Path to a file containing a set of prefixes to prepend to the texts to be decoded")
    parser.add_argument("--decoded_texts_save_path", type=str, default=None, help="Path to save the decoded texts to. Must specify a single file.")
    parser.add_argument("--decoded_texts_load_path", type=str, default=None, help="Path to load the decoded texts from. Must specify a single file.")
    parser.add_argument("--loaded_texts_subsample", type=int, default=None, help="If specified, will randomly subsample the loaded decoded texts to this number. Take care not to accidentally mess up the correspondence between the decoded texts and the loaded embeddings. Recompute the embeddings if in doubt.")
    parser.add_argument("--base_model_quant_level", type=str, default="8bit", choices=["4bit", "8bit", "bfloat16"], help="Quantization level of the base model")
    parser.add_argument("--intervention_model_quant_level", type=str, default="8bit", choices=["4bit", "8bit", "bfloat16"], help="Quantization level of the intervention model")
    parser.add_argument("--num_decodings_per_prompt", type=int, default=None, help="Number of decodings per prompt to use for label generation. If not specified, all decodings will be used.")
    parser.add_argument("--include_prompts_in_decoded_texts", action="store_true", help="Flag to include the prompts in the decoded texts")
    parser.add_argument("--single_cluster_label_instruction", type=str, default=None, help="Instructions for generating the single cluster labels")
    parser.add_argument("--contrastive_cluster_label_instruction", type=str, default=None, help="Instructions for generating the contrastive cluster labels")
    parser.add_argument("--diversify_contrastive_labels", action="store_true", help="Flag to diversify the contrastive labels by clustering the previously generated labels, and then using the assistant to summarize the common themes across the labels closest to the cluster centers. Then we provide those summaries to the assistant to generate new labels that are different from the previous ones.")
    parser.add_argument("--verified_diversity_promoter", action="store_true", help="Flag to promote diversity in the contrastive labels by recording any hypotheses that are verified discriminatively, providing them to the assistant, and asking the assistant to look for other hypotheses that are different.")
    parser.add_argument("--run_weight_analysis", action="store_true", help="Flag to run the weight analysis looking at how the weights of the base model and intervention model differ across all named parameters")


    ## Clustering
    parser.add_argument("--cluster_method", type=str, default="kmeans", help="Method to use for clustering. Options: kmeans, hdbscan")
    parser.add_argument("--num_clusters", type=int, default=40, help="Number of clusters to use for clustering")
    parser.add_argument("--min_cluster_size", type=int, default=7, help="Minimum cluster size to use for clustering. Only matters when using HDBSCAN")
    parser.add_argument("--max_cluster_size", type=int, default=2000, help="Maximum cluster size to use for clustering. Only matters when using HDBSCAN")
    parser.add_argument("--sampled_comparison_texts_per_cluster", type=int, default=50, help="Number of texts to sample for each cluster when validating the contrastive discriminative score of labels between neighboring clusters")
    parser.add_argument("--cross_validate_contrastive_labels", action="store_true", help="Flag to cross-validate the contrastive labels by testing the discriminative score of the labels on different clusters from which they were generated.")
    parser.add_argument("--sampled_texts_per_cluster", type=int, default=10, help="Number of texts to sample for each cluster when generating labels")
    parser.add_argument("--clustering_instructions", type=str, default="Identify the topic or theme of the given texts", help="Instructions provided to the local embedding model for clustering")
    parser.add_argument("--n_clustering_inits", type=int, default=10, help="Number of clustering initializations to use for clustering")
    parser.add_argument("--use_prompts_as_clusters", action="store_true", help="Flag to use the prompts to determine the clusters")
    parser.add_argument("--cluster_on_prompts", action="store_true", help="Flag to perform clustering on the prompts, then use those to determine the clusters")
    parser.add_argument("--local_embedding_model_str", type=str, default="intfloat/multilingual-e5-large-instruct", help="Model version for the local embedding model")
    parser.add_argument("--K", type=int, default=3, help="Number of neighbors to connect each cluster to")
    parser.add_argument("--match_by_ids", action="store_true", help="Flag to match clusters by their IDs, not embedding distances.")

    # Validation
    parser.add_argument("--num_rephrases_for_validation", type=int, default=0, help="Number of rephrases of each generated hypothesis to generate for validation")
    parser.add_argument("--num_generated_texts_per_description", type=int, default=20, help="Number of generated texts per description for generative validation")
    parser.add_argument("--generated_labels_per_cluster", type=int, default=3, help="Number of generated labels to generate for each cluster")

    parser.add_argument("--use_unitary_comparisons", action="store_true", help="Flag to use unitary comparisons")
    parser.add_argument("--max_unitary_comparisons_per_label", type=int, default=100, help="Maximum number of unitary comparisons to perform per label")
    parser.add_argument("--additional_unitary_comparisons_per_label", type=int, default=0, help="Additional number of unitary comparisons to perform per label")
    parser.add_argument("--match_cutoff", type=float, default=0.69, help="Accuracy / AUC cutoff for determining matching/unmatching clusters")
    parser.add_argument("--discriminative_query_rounds", type=int, default=3, help="Number of rounds of discriminative queries to perform")
    parser.add_argument("--discriminative_validation_runs", type=int, default=5, help="Number of validation runs to perform for each model for each hypothesis")
    parser.add_argument("--metric", type=str, default="acc", choices=["acc", "auc"], help="Metric to use for validation of labels")
    # t-SNE
    parser.add_argument("--tsne_save_path", type=str, default=None, help="Path to save the t-SNE plot to")
    parser.add_argument("--tsne_title", type=str, default=None, help="Title for the t-SNE plot")
    parser.add_argument("--tsne_perplexity", type=int, default=30, help="Perplexity parameter for t-SNE")

    # API provider
    parser.add_argument("--api_provider", type=str, choices=["anthropic", "openai", "gemini"], required=True, help="API provider for ground truth generation and comparison")
    parser.add_argument("--model_str", type=str, required=True, help="Model version for the chosen API provider")
    parser.add_argument("--stronger_model_str", type=str, default=None, help="Model version for an optional second model more capable than the one indicated with model_str; can be used for key steps that are not repeated often")
    parser.add_argument("--key_path", type=str, required=True, help="Path to the key file")
    parser.add_argument("--openrouter_api_key_path", type=str, default=None, help="Path to the openrouter API key file")
    parser.add_argument("--api_interactions_save_loc", type=str, default=None, help="File location to record any API model interactions. Defaults to None and no recording of interactions.")

    args = parser.parse_args()
    main(args)