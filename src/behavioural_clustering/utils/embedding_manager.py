import json
import os
from typing import List
import numpy as np
from behavioural_clustering.config.run_settings import EmbeddingSettings
from behavioural_clustering.utils.embedding_utils import embed_texts

class EmbeddingManager:
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.embeddings_file = os.path.join(storage_path, "statement_embeddings.json")
        self.load_embeddings()

    def load_embeddings(self):
        if os.path.exists(self.embeddings_file):
            with open(self.embeddings_file, 'r') as f:
                self.embeddings_data = json.load(f)
        else:
            self.embeddings_data = {}

    def save_embeddings(self):
        with open(self.embeddings_file, 'w') as f:
            json.dump(self.embeddings_data, f)

    def get_or_create_embeddings(self, statements: List[str], embedding_settings: EmbeddingSettings) -> List[np.ndarray]:
        embeddings = []
        new_statements = []
        new_embeddings_indices = []

        for i, statement in enumerate(statements):
            embedding_key = self._get_embedding_key(statement, embedding_settings)
            if embedding_key in self.embeddings_data:
                embeddings.append(np.array(self.embeddings_data[embedding_key]["embedding"]))
            else:
                new_statements.append(statement)
                new_embeddings_indices.append(i)

        if new_statements:
            print(f"Creating {len(new_statements)} new embeddings")
            new_embeddings = embed_texts(new_statements, embedding_settings)
            print(f"Type of new_embeddings: {type(new_embeddings)}")
            print(f"Type of first new embedding: {type(new_embeddings[0])}")
            
            for i, embedding in zip(new_embeddings_indices, new_embeddings):
                statement = statements[i]
                embedding_key = self._get_embedding_key(statement, embedding_settings)
                self.embeddings_data[embedding_key] = {
                    "statement": statement,
                    "embedding": embedding.tolist(),  # Store as list in JSON
                    "embedding_model": embedding_settings.embedding_model,
                    "other_params": embedding_settings.other_params
                }
                embeddings.insert(i, embedding)  # Insert as numpy array

            self.save_embeddings()

        print(f"Final number of embeddings: {len(embeddings)}")
        print(f"Type of first final embedding: {type(embeddings[0])}")
        return embeddings

    def _get_embedding_key(self, statement: str, embedding_settings: EmbeddingSettings) -> str:
        return f"{statement}_{embedding_settings.embedding_model}"