from typing import List
from tqdm import tqdm
from openai import OpenAI
import time
import numpy as np

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

        embeddings += [np.array(item.embedding) for item in embeddings_data]
    
    print(f"Number of texts to embed: {len(texts)}")
    print(f"Number of embeddings created: {len(embeddings)}")
    print(f"Type of first embedding: {type(embeddings[0])}")
    print(f"Shape of first embedding: {embeddings[0].shape}")
    
    return embeddings