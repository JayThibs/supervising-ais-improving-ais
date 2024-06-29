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
                print(texts)
                text_subset = texts[start_idx:end_idx]
                print("text_subset:", text_subset)
                embeddings_data = client.embeddings.create(
                    embedding_model=embedding_model, input=text_subset
                ).data

                break  # Exit the retry loop if successful
            except Exception as e:
                print(f"Skipping due to server error number {retry_count}: {e}")
                time.sleep(initial_sleep_time * (2**retry_count))  # Exponential backoff

        # print("embedding:", embedding)
        embeddings += [item.embedding for item in embeddings_data]
    return embeddings