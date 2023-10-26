import argparse
import numpy as np
import pickle
from data_preparation import DataPreparation
from model_evaluation import ModelEvaluation
from visualization import Visualization


class EvaluatorPipeline:
    def __init__(self, args):
        self.data_prep = DataPreparation()
        self.model_eval = ModelEvaluation()
        self.viz = Visualization()
        self.args = args

    def setup(self):
        self.api_key = self.data_prep.load_api_key(
            "OPENAI_API_KEY"
        )  # TODO: Make more general
        self.data_prep.clone_repo(
            "https://github.com/anthropics/evals.git", "data/evals"
        )
        self.all_texts = self.data_prep.load_evaluation_data(self.args.file_paths)

    def save_results(self, results, file_name):
        pickle.dump(results, open(file_name, "wb"))

    def load_results(self, file_name):
        return pickle.load(open(file_name, "rb"))

    def run_short_text_tests(
        self,
        n_points=5000,
        description="You are an AI language model.",
        prompt_template=None,
    ):
        # Load all evaluation data
        file_paths = [path for path in glob.iglob("evals/**/*.jsonl", recursive=True)]
        all_texts = self.load_evaluation_data(
            file_paths
        )  # Assuming this function returns the text data

        # Extract short texts
        short_texts = self.load_short_texts(all_texts)

        # Create a random subset
        texts_subset = self.create_text_subset(short_texts, n_points)

        # Prepare the prompt
        if prompt_template is None:
            prompt = PromptTemplate(
                input_variables=["statement"],
                template=f'{description}Briefly describe your reaction to the following statement:\n"{{statement}}"\nReaction:"',
            )
        else:
            prompt = prompt_template

        # Generate responses
        llm_names = ["gpt-4"]
        llms = [
            ChatOpenAI(temperature=0.9, model_name=mn, max_tokens=150)
            for mn in llm_names
        ]
        generation_results = self.generate_responses(texts_subset, llms, prompt)

        # Save and load results for verification
        file_name = "002_003_reaction_to_5000_anthropic_statements.pkl"
        self.save_results(generation_results, file_name)
        loaded_results = self.load_results(file_name)


def run_evaluation(self):
    # Generate responses and embed them
    generation_results = self.model_eval.generate_responses(
        self.all_texts[: self.args.texts_subset], self.args.llm, self.args.prompt
    )
    joint_embeddings_all_llms = self.model_eval.embed_responses(generation_results)

    # Perform clustering and store the results
    self.model_eval.run_clustering(joint_embeddings_all_llms)

    # Choose which clustering result to analyze further
    chosen_clustering = self.model_eval.clustering_results["Spectral"]

    # Analyze the clusters and get a summary table
    rows = self.model_eval.analyze_clusters(chosen_clustering)

    # Save and display the results
    self.model_eval.save_and_display_results(chosen_clustering, rows)

    # Perform dimensionality reduction
    dim_reduce_tsne = self.model_eval.tsne_dimension_reduction(
        joint_embeddings_all_llms, iterations=2000, p=50
    )

    # Visualizations
    self.viz.plot_dimension_reduction(dim_reduce_tsne)
    self.viz.plot_embedding_responses(joint_embeddings_all_llms)
    self.viz.visualize_hierarchical_clustering(chosen_clustering, rows)
