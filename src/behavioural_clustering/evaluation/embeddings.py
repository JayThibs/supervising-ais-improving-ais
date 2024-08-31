from typing import List, Dict, Any
import numpy as np
from behavioural_clustering.config.run_settings import EmbeddingSettings
from behavioural_clustering.utils.embedding_utils import embed_texts

def create_embeddings(query_results_per_model, llms, embedding_settings: EmbeddingSettings, embedding_manager):
    print(f"Starting create_embeddings method")
    print(f"Number of models: {len(llms)}")
    print(f"Number of query results: {len(query_results_per_model)}")

    joint_embeddings_all_llms = []
    for model_num, (model_family, model_name) in enumerate(llms):
        print(f"Processing model {model_num}: {model_name}")
        
        if model_num >= len(query_results_per_model):
            print(f"Warning: No query results for model {model_name}")
            continue

        model_results = query_results_per_model[model_num]
        
        if "inputs" not in model_results or "responses" not in model_results:
            print(f"Warning: Missing 'inputs' or 'responses' for model {model_name}")
            continue

        inputs = model_results["inputs"]
        responses = model_results["responses"]
        
        print(f"Number of inputs: {len(inputs)}")
        print(f"Number of responses: {len(responses)}")

        inputs_embeddings = embedding_manager.get_or_create_embeddings(inputs, embedding_settings)
        print(f"Number of input embeddings: {len(inputs_embeddings)}")

        responses_embeddings = embedding_manager.get_or_create_embeddings(responses, embedding_settings)
        print(f"Number of response embeddings: {len(responses_embeddings)}")

        joint_embeddings = [np.concatenate([inp, r]) for inp, r in zip(inputs_embeddings, responses_embeddings)]
        print(f"Number of joint embeddings: {len(joint_embeddings)}")

        for input, response, embedding in zip(inputs, responses, joint_embeddings):
            joint_embeddings_all_llms.append({
                "model_num": model_num,
                "statement": input,
                "response": response,
                "embedding": embedding,  # Store as numpy array
                "model_name": model_name
            })

    print(f"Total number of joint embeddings: {len(joint_embeddings_all_llms)}")

    if not joint_embeddings_all_llms:
        print("Warning: No joint embeddings were created")
        return []

    return joint_embeddings_all_llms