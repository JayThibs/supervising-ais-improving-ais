import csv
import numpy as np
import pdb
from tqdm import tqdm
from typing import List, Tuple
from sklearn.cluster import KMeans
from prettytable import PrettyTable
from behavioural_clustering.models.model_factory import initialize_model
from behavioural_clustering.config.run_settings import RunSettings
from behavioural_clustering.utils.model_utils import query_model_on_statements


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

        model_info = {
            "model_family": model_family,
            "model": model,
            "system_message": system_message
        }
        model_instance = initialize_model(model_info, temperature=0, max_tokens=5)

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