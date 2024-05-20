import os
import csv
import numpy as np
import pdb
from tqdm import tqdm
from typing import List, Tuple
from utils import *
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from prettytable import PrettyTable
from models import OpenAIModel, AnthropicModel, LocalModel
from config.run_settings import RunSettings


class ModelEvaluation:
    def __init__(self, run_settings: RunSettings, llms: List[Tuple[str, str]]):
        self.settings = run_settings
        self.llms = llms

    def get_model_approvals(
        self,
        statements,
        prompt_template,
        model_family,
        model,
        system_message="",
        approve_strs=["yes"],
        disapprove_strs=["no"],
    ) -> list:
        approvals = []
        n_statements = len(statements)
        prompts = [prompt_template.format(statement=s) for s in statements]
        model_instance = None

        if model_family == "openai":
            print("System message:", system_message)
            print("Model:", model)
            model_instance = OpenAIModel(
                model, system_message, temperature=0, max_tokens=5
            )
        elif model_family == "anthropic":
            model_instance = AnthropicModel(
                model, system_message, temperature=0, max_tokens=5
            )
        elif model_family == "local":
            model_instance = LocalModel()
        else:
            raise ValueError(
                "Invalid model_family name. Choose 'openai', 'anthropic', or 'local'. Common error: it's a list or capitalization issue."
            )

        for i in tqdm(range(n_statements)):
            print(f"Prompt {i}: {prompts[i]}")
            r = model_instance.generate(prompt=prompts[i]).lower()

            approve_strs_in_response = sum([s in r for s in approve_strs])
            disapprove_strs_in_response = sum([s in r for s in disapprove_strs])

            if approve_strs_in_response and not disapprove_strs_in_response:
                approvals.append(1)
            elif not approve_strs_in_response and disapprove_strs_in_response:
                approvals.append(0)
            else:
                # Uncertain response:
                approvals.append(-1)

        return approvals

    def tsne_dimension_reduction(self, embeddings):
        """
        Performs dimensionality reduction on embeddings using t-SNE.

        Parameters:
        embeddings (list): A list of embeddings to reduce.
        dimensions (int): The number of dimensions to reduce to. Default is 2.
        perplexity (int): The perplexity parameter for t-SNE. Default is 50.
        iterations (int): The number of iterations for optimization. Default is 2000.
        random_state (int): The seed for random number generator. Default is 42.

        Returns:
        np.ndarray: The reduced embeddings as a NumPy array.
        """
        # Perform the t-SNE dimensionality reduction
        tsne_settings = self.settings.tsne_settings
        tsne = TSNE(
            n_components=tsne_settings.dimensions,
            perplexity=tsne_settings.perplexity,
            n_iter=tsne_settings.n_iter,
            angle=tsne_settings.angle,
            init=tsne_settings.init,
            early_exaggeration=tsne_settings.early_exaggeration,
            learning_rate=tsne_settings.learning_rate,
            random_state=self.settings.random_state,
        )
        reduced_embeddings = tsne.fit_transform(X=embeddings)

        return reduced_embeddings

    def perform_clustering(
        self, combined_embeddings: np.array, n_clusters: int = 200
    ) -> KMeans:
        """Perform clustering on combined embeddings."""
        clustering_settings = self.settings.clustering_settings
        clustering = KMeans(
            n_clusters=clustering_settings.n_clusters,
            random_state=self.settings.random_state,
        ).fit(combined_embeddings)
        return clustering

    def run_eval(
        self,
        text_subset,
        n_statements,
        llm,
    ):
        prompt_settings = self.settings.prompt_settings
        data_settings = self.settings.data_settings
        system_message = prompt_settings.statements_system_message
        prompt_template = prompt_settings.statements_prompt_template
        model_family, model = llm[0], llm[1]

        # Check if there is a saved generated responses file
        file_name = f"{model_family}_{model}_reaction_to_{data_settings.n_statements}_anthropic_statements.pkl"
        query_results = query_model_on_statements(
            text_subset, model_family, model, prompt_template, system_message
        )  # dictionary of inputs, responses, and model instance
        return query_results, file_name

    def display_statement_themes(self, chosen_clustering, rows, model_info_list):
        print(f"Chosen clustering: {chosen_clustering}")
        print(f"Rows: {rows}")
        # Create a table and save it in a readable format (CSV) for easy visualization in VSCode
        model_columns = [
            model_info["model"] for model_info in model_info_list
        ]  # Extract model names from model_info_list
        table_headers = (
            [
                "ID",  # cluster ID
                "N",  # number of items in the cluster
            ]
            + model_columns
            + [  # Add model names dynamically
                "Inputs Themes",  # LLM says the theme of the input
                "Responses Themes",  # LLM says the theme of the response
                "Interaction Themes",  # LLM says the theme of the input and response together
            ]
        )
        plot_settings = self.settings.plot_settings
        csv_file_path = f"{plot_settings.save_path}/tables/cluster_results_table_statement_responses.csv"
        with open(csv_file_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(table_headers)
            writer.writerows(rows)

        # Display the table in the console
        t = PrettyTable()
        t.field_names = table_headers
        for row in rows:
            t.add_row(row)
        print(t)

    def run_evaluation(self, data):
        pass
