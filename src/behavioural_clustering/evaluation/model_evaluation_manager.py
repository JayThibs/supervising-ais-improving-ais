import numpy as np
from typing import List, Tuple, Dict
from behavioural_clustering.config.run_settings import RunSettings
from behavioural_clustering.utils.model_utils import query_model_on_statements
from behavioural_clustering.evaluation.embeddings import embed_texts

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
        joint_embeddings_all_llms = []
        for model_num, (model_family, model) in enumerate(llms):
            inputs = query_results_per_model[model_num]["inputs"]
            responses = query_results_per_model[model_num]["responses"]
            
            if model_num == 0:
                inputs_embeddings = embed_texts(texts=inputs, embedding_settings=embedding_settings)
            
            responses_embeddings = embed_texts(texts=responses, embedding_settings=embedding_settings)
            joint_embeddings = [inp + r for inp, r in zip(inputs_embeddings, responses_embeddings)]
            
            for input, response, embedding in zip(inputs, responses, joint_embeddings):
                joint_embeddings_all_llms.append([model_num, input, response, embedding, model])

        combined_embeddings = np.array([e[3] for e in joint_embeddings_all_llms])
        return joint_embeddings_all_llms, combined_embeddings