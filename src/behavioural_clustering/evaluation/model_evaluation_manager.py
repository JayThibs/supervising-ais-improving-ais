import numpy as np
from typing import List, Tuple, Dict
from behavioural_clustering.config.run_settings import RunSettings
from behavioural_clustering.utils.model_utils import query_model_on_statements
from behavioural_clustering.evaluation.embeddings import embed_texts
from behavioural_clustering.models.model_factory import initialize_model
from tqdm import tqdm

class ModelEvaluationManager:
    def __init__(self, run_settings: RunSettings, llms: List[Tuple[str, str]]):
        self.settings = run_settings
        self.llms = llms
        self.model_info_list = [{"model_family": family, "model": model} for family, model in llms]

    def generate_responses(self, text_subset):
        query_results_per_model = []
        for model_family, model in self.llms:
            query_results = query_model_on_statements(
                text_subset,
                model_family,
                model,
                self.settings.prompt_settings.statements_prompt_template,
                self.settings.prompt_settings.statements_system_message
            )
            query_results_per_model.append(query_results)
        return query_results_per_model

    def create_embeddings(self, query_results_per_model, llms, embedding_settings):
        print(f"Starting create_embeddings method")
        print(f"Number of models: {len(llms)}")
        print(f"Number of query results: {len(query_results_per_model)}")

        joint_embeddings_all_llms = []
        for model_num, (model_family, model) in enumerate(llms):
            print(f"Processing model {model_num}: {model}")
            
            if model_num >= len(query_results_per_model):
                print(f"Warning: No query results for model {model}")
                continue

            model_results = query_results_per_model[model_num]
            
            if "inputs" not in model_results or "responses" not in model_results:
                print(f"Warning: Missing 'inputs' or 'responses' for model {model}")
                continue

            inputs = model_results["inputs"]
            responses = model_results["responses"]
            
            print(f"Number of inputs: {len(inputs)}")
            print(f"Number of responses: {len(responses)}")

            if model_num == 0:
                inputs_embeddings = embed_texts(texts=inputs, embedding_settings=embedding_settings)
                print(f"Number of input embeddings: {len(inputs_embeddings)}")

            responses_embeddings = embed_texts(texts=responses, embedding_settings=embedding_settings)
            print(f"Number of response embeddings: {len(responses_embeddings)}")

            joint_embeddings = [inp + r for inp, r in zip(inputs_embeddings, responses_embeddings)]
            print(f"Number of joint embeddings: {len(joint_embeddings)}")

            for input, response, embedding in zip(inputs, responses, joint_embeddings):
                joint_embeddings_all_llms.append([model_num, input, response, embedding, model])

        print(f"Total number of joint embeddings: {len(joint_embeddings_all_llms)}")

        if not joint_embeddings_all_llms:
            print("Warning: No joint embeddings were created")
            return [], []

        combined_embeddings = [e[3] for e in joint_embeddings_all_llms]
        print(f"Number of combined embeddings: {len(combined_embeddings)}")

        return joint_embeddings_all_llms, combined_embeddings

    def get_model_approvals(
        self,
        statements: List[str],
        prompt_template: str,
        model_family: str,
        model: str,
        approve_strs: List[str] = ["yes"],
        disapprove_strs: List[str] = ["no"],
    ) -> List[int]:
        approvals = []
        n_statements = len(statements)
        prompts = [prompt_template.format(statement=s) for s in statements]

        model_info = {
            "model_family": model_family,
            "model": model,
        }
        model_instance = initialize_model(model_info, temperature=0, max_tokens=5)

        for i, prompt in enumerate(prompts):
            print(f"Prompt {i+1}/{n_statements}: {prompt}")
            r = model_instance.generate(prompt=prompt).lower()

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