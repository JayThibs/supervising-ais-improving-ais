import argparse
import pandas as pd
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_finetuning_data import generate_ground_truths, generate_dataset
from auto_finetuning_compare_to_truth import compare_and_score_hypotheses
from auto_finetuning_helpers import load_api_key, parse_dict
from auto_finetuning_train import dummy_finetune_model, finetune_model
from auto_finetuning_interp import dummy_apply_interpretability_method, apply_interpretability_method

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
        self.base_model = None
        self.tokenizer = None
        self.finetuned_model = None
        self.ground_truths = None
        self.ground_truths_df = None
        self.dummy_finetune = True if args.dummy_finetune else False
        self.dummy_interp = True if args.dummy_interp else False


    def load_ground_truths_and_data(self):
        """Load ground truths and associated data from the CSV file."""
        self.ground_truths_df = pd.read_csv(self.args.output_file_path)
        print("ground_truths_df", self.ground_truths_df)

    def load_base_model(self):
        """Load the base model and tokenizer."""
        self.base_model = AutoModelForCausalLM.from_pretrained(self.args.base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.base_model)

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
        if self.args.load_ground_truths:
            self.load_ground_truths_and_data()
        else:
            self.ground_truths = generate_ground_truths(
                self.args.num_ground_truths,
                self.args.api_provider,
                self.args.model_str,
                self.key,   
                self.args.focus_area
            )
            print("ground_truths", self.ground_truths)
            self.ground_truths_df = generate_dataset(
                self.ground_truths,
                self.args.num_samples,
                self.args.api_provider,
                self.args.model_str,
                self.key,
                self.args.output_file_path
            )
            print("ground_truths_df", self.ground_truths_df)
        
        self.load_base_model()
        self.finetune_model()
        if self.dummy_interp:
            discovered_hypotheses = dummy_apply_interpretability_method(self.base_model, self.finetuned_model)
        else:
            '''
            base_model: PreTrainedModel, 
            finetuned_model: PreTrainedModel, 
            n_decoded_texts: int = 2000, 
            decoding_prefix_file: Optional[str] = None, 
            api_provider: str = "anthropic",
            api_model_str: str = "claude-3-haiku-20240307",
            auth_key: Optional[str] = None,
            client: Optional[Any] = None,
            local_embedding_model_str: Optional[str] = None, 
            local_embedding_api_key: Optional[str] = None,
            init_clustering_from_base_model: bool = False,
            clustering_instructions: str = "Identify the topic or theme of the given texts",
            device: str = "cuda:0",
            cluster_method: str = "kmeans",
            n_clusters: int = 30,
            min_cluster_size: int = 7,
            max_cluster_size: int = 2000
            '''
            discovered_hypotheses = apply_interpretability_method(
                self.base_model, 
                self.finetuned_model,
                self.tokenizer,
                n_decoded_texts=2000,
                decoding_prefix_file=None,
                api_provider=self.args.api_provider,
                api_model_str=self.args.model_str,
                auth_key=self.key,
                client=None,
                local_embedding_model_str="nvidia/NV-Embed-v1",
                local_embedding_api_key=None,
                init_clustering_from_base_model=True,
                clustering_instructions="Identify the topic or theme of the given texts",
                device="cuda:0",
                cluster_method="kmeans",
                n_clusters=30,
                min_cluster_size=7,
                max_cluster_size=2000
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
    parser.add_argument("--base_model", type=str, required=True, help="HuggingFace model ID for the base model")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of data points to generate per ground truth (around 10-1000)")
    parser.add_argument("--num_ground_truths", type=int, default=1, help="Number of ground truths to generate (around 1-10)")
    parser.add_argument("--api_provider", type=str, choices=["anthropic", "openai"], required=True, help="API provider for ground truth generation and comparison")
    parser.add_argument("--model_str", type=str, required=True, help="Model version for the chosen API provider")
    parser.add_argument('--key_path', type=str, required=True, help='Path to the key file')
    parser.add_argument("--output_file_path", type=str, required=True, help="Path to save the generated CSV file")
    parser.add_argument("--focus_area", type=str, default=None, help="Optional focus area for ground truth generation")
    parser.add_argument("--dummy_finetune", action="store_true", help="Flag to use dummy finetuning")
    parser.add_argument("--dummy_interp", action="store_true", help="Flag to use dummy interpretability method")
    parser.add_argument("--finetuning_params", type=parse_dict, default="{}", help="Parameters for finetuning as a JSON string")
    parser.add_argument("--load_ground_truths", action="store_true", help="Flag to load ground truths from a file")

    args = parser.parse_args()
    main(args)