from typing import List, Dict, Tuple, Any, Optional, Iterator, Union
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

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
        
        Args:
            models: List of (model_family, model_name) tuples
        """
        self.model_order = models
        self.embeddings: List[EmbeddingEntry] = []
        self._validate_models()
        
    def __iter__(self) -> Iterator[EmbeddingEntry]:
        """
        Make JointEmbeddings iterable to simplify code that uses it.
        
        Returns:
            Iterator over embedding entries
        """
        return iter(self.embeddings)
        
    def __len__(self) -> int:
        """
        Get the number of embedding entries.
        
        Returns:
            Number of embedding entries
        """
        return len(self.embeddings)

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
                     embedding: np.ndarray) -> None:
        """
        Add an embedding entry while maintaining order.
        
        Args:
            model_idx: Index of the model in the model_order list
            statement: The statement/prompt that was given to the model
            response: The model's response to the statement
            embedding: The embedding vector for the response
            
        Raises:
            ValueError: If model_idx is out of range
            ValueError: If embedding is not a numpy array
        """
        if model_idx >= len(self.model_order):
            raise ValueError(f"Model index {model_idx} out of range")
        
        if not isinstance(embedding, np.ndarray):
            try:
                embedding = np.array(embedding)
                logger.warning(f"Converting embedding to numpy array for model {model_idx}")
            except Exception as e:
                raise ValueError(f"Could not convert embedding to numpy array: {str(e)}")
            
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

    def get_embedding_matrix(self, response_only: bool = False) -> np.ndarray:
        """
        Get embeddings as a matrix.
        
        Args:
            response_only: If True, return only the response part of the embedding
                          (assumes embeddings are concatenated [input, response])
        
        Returns:
            numpy.ndarray: Matrix of embeddings
            
        Raises:
            ValueError: If no embeddings are available
        """
        if not self.embeddings:
            raise ValueError("No embeddings available")
        
        try:
            if response_only:
                embeddings = []
                for e in self.embeddings:
                    embedding_dim = e.embedding.shape[0] // 2
                    embeddings.append(e.embedding[embedding_dim:])
                return np.array(embeddings)
            else:
                return np.array([e.embedding for e in self.embeddings])
        except Exception as e:
            logger.error(f"Error creating embedding matrix: {str(e)}")
            fixed_embeddings = []
            for i, entry in enumerate(self.embeddings):
                if not isinstance(entry.embedding, np.ndarray):
                    try:
                        fixed_embeddings.append(np.array(entry.embedding))
                        logger.warning(f"Fixed embedding at index {i}")
                    except:
                        logger.error(f"Could not fix embedding at index {i}")
                        fixed_embeddings.append(np.zeros(self.embeddings[0].embedding.shape))
                else:
                    fixed_embeddings.append(entry.embedding)
            
            return np.array(fixed_embeddings)

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
        
        Returns:
            bool: True if all statements have embeddings from all models, False otherwise
        """
        if not self.embeddings:
            logger.warning("No embeddings available for validation")
            return False
            
        # Group by statement
        statement_groups: Dict[str, List[int]] = {}
        for entry in self.embeddings:
            if entry.statement not in statement_groups:
                statement_groups[entry.statement] = []
            statement_groups[entry.statement].append(entry.model_idx)
        
        # Check each statement has embeddings from all models
        expected_models = set(range(len(self.model_order)))
        
        missing_combinations = []
        for statement, models in statement_groups.items():
            missing_models = expected_models - set(models)
            if missing_models:
                missing_combinations.append((statement, missing_models))
        
        if missing_combinations:
            for statement, missing_models in missing_combinations:
                model_names = [self.model_order[idx][1] for idx in missing_models]
                logger.warning(f"Statement '{statement[:50]}...' is missing embeddings from models: {model_names}")
            return False
            
        return True            