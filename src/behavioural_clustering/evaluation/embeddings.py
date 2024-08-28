from typing import List
from tqdm import tqdm
from openai import OpenAI
import time

def embed_texts(
    texts: List[str],
    embedding_settings,
):
    embedding_model, batch_size, max_retries, initial_sleep_time = (
        embedding_settings.embedding_model,
        embedding_settings.batch_size,
        embedding_settings.max_retries,
        embedding_settings.initial_sleep_time,
    )
    client = OpenAI()
    embeddings = []
    n_texts = len(texts)
    n_batches = n_texts // batch_size + int(n_texts % batch_size != 0)

    for i in tqdm(range(n_batches)):
        for retry_count in range(max_retries):
            try:
                start_idx = batch_size * i
                end_idx = min(batch_size * (i + 1), n_texts)
                text_subset = texts[start_idx:end_idx]
                embeddings_data = client.embeddings.create(
                    model=embedding_model, input=text_subset
                )
                embeddings_data = embeddings_data.data

                break  # Exit the retry loop if successful
            except TypeError as te:
                print(f"TypeError encountered: {te}")
                break  # Exit the retry loop if there is a TypeError
            except Exception as e:
                print(f"Skipping due to server error number {retry_count}: {e}")
                time.sleep(initial_sleep_time * (2**retry_count))  # Exponential backoff

        embeddings += [item.embedding for item in embeddings_data]
    
    print(f"Number of texts to embed: {len(texts)}")
    print(f"Number of embeddings created: {len(embeddings)}")
    return embeddings

def create_embeddings(query_results_per_model, llms, embedding_settings):
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

        if model_num == 0:
            inputs_embeddings = embed_texts(texts=inputs, embedding_settings=embedding_settings)
            print(f"Number of input embeddings: {len(inputs_embeddings)}")

        responses_embeddings = embed_texts(texts=responses, embedding_settings=embedding_settings)
        print(f"Number of response embeddings: {len(responses_embeddings)}")

        joint_embeddings = [inp + r for inp, r in zip(inputs_embeddings, responses_embeddings)]
        print(f"Number of joint embeddings: {len(joint_embeddings)}")

        for input, response, embedding in zip(inputs, responses, joint_embeddings):
            joint_embeddings_all_llms.append({"model_num": model_num, "statement": input, "response": response, "embedding": embedding, "model_name": model_name})

    print(f"Total number of joint embeddings: {len(joint_embeddings_all_llms)}")

    if not joint_embeddings_all_llms:
        print("Warning: No joint embeddings were created")
        return []

    return joint_embeddings_all_llms