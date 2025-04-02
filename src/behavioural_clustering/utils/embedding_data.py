from typing import List, Dict, Tuple, Any
import numpy as np
from dataclasses import dataclass

@dataclass
class EmbeddingEntry:
    model_idx: int
    model_family: str
    model_name: str
    statement: str
    response: str
    embedding: np.ndarray

    def to_dict(self) -> Dict:
        return {
            "model_idx": self.model_idx,
            "model_family": self.model_family,
            "model_name": self.model_name,
            "statement": self.statement,
            "response": self.response,
            "embedding": self.embedding.tolist()
        }

    @staticmethod
    def from_dict(data: Dict) -> 'EmbeddingEntry':
        return EmbeddingEntry(
            model_idx=data["model_idx"],
            model_family=data["model_family"],
            model_name=data["model_name"],
            statement=data["statement"],
            response=data["response"],
            embedding=np.array(data["embedding"])
        )

class JointEmbeddings:
    def __init__(self, models: List[Tuple[str, str]]):
        """
        Initialize with list of (model_family, model_name) tuples.
        The order of models is critical and must be maintained.
        """
        self.model_order = models
        self.embeddings: List[EmbeddingEntry] = []
        self._validate_models()

    def _validate_models(self):
        """Ensure model list is valid."""
        if not self.model_order:
            raise ValueError("Must provide at least one model")
        if not all(isinstance(m, tuple) and len(m) == 2 for m in self.model_order):
            raise ValueError("Each model must be a (family, name) tuple")

    def add_embedding(self, 
                     model_idx: int, 
                     statement: str, 
                     response: str, 
                     embedding: np.ndarray):
        """
        Add an embedding entry while maintaining order.
        """
        if model_idx >= len(self.model_order):
            raise ValueError(f"Model index {model_idx} out of range")
            
        model_family, model_name = self.model_order[model_idx]
        entry = EmbeddingEntry(
            model_idx=model_idx,
            model_family=model_family,
            model_name=model_name,
            statement=statement,
            response=response,
            embedding=embedding
        )
        self.embeddings.append(entry)

    def get_embeddings_by_model(self, model_idx: int) -> List[EmbeddingEntry]:
        """Get all embeddings for a specific model."""
        return [e for e in self.embeddings if e.model_idx == model_idx]

    def get_all_embeddings(self) -> List[EmbeddingEntry]:
        """Get all embeddings in their original order."""
        return self.embeddings

    def get_embedding_matrix(self) -> np.ndarray:
        """Get embeddings as a matrix."""
        if not self.embeddings:
            raise ValueError("No embeddings available")
        return np.array([e.embedding for e in self.embeddings])

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "model_order": self.model_order,
            "embeddings": [e.to_dict() for e in self.embeddings]
        }

    @staticmethod
    def from_dict(data: Dict) -> 'JointEmbeddings':
        """Create from dictionary."""
        joint_embeddings = JointEmbeddings(data["model_order"])
        for embedding_data in data["embeddings"]:
            entry = EmbeddingEntry.from_dict(embedding_data)
            joint_embeddings.embeddings.append(entry)
        return joint_embeddings

    def validate_completeness(self) -> bool:
        """
        Verify that we have embeddings from all models for all statements.
        """
        if not self.embeddings:
            return False
            
        # Group by statement
        statement_groups: Dict[str, List[int]] = {}
        for entry in self.embeddings:
            if entry.statement not in statement_groups:
                statement_groups[entry.statement] = []
            statement_groups[entry.statement].append(entry.model_idx)
            
        # Check each statement has embeddings from all models
        expected_models = set(range(len(self.model_order)))
        return all(set(models) == expected_models 
                  for models in statement_groups.values()) 