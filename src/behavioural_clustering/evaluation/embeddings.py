from typing import List, Dict, Any
import numpy as np
from behavioural_clustering.config.run_settings import EmbeddingSettings
from behavioural_clustering.utils.embedding_utils import embed_texts
from behavioural_clustering.utils.cache_management import CacheManager, CacheMetadata, ModelParams, EmbeddingParams, DataParams
from behavioural_clustering.utils.embedding_data import JointEmbeddings
import logging

logger = logging.getLogger(__name__)

def create_embeddings(query_results: Dict[str, Any], 
                     llms: List[tuple[str, str]], 
                     embedding_settings: EmbeddingSettings,
                     embedding_manager,
                     cache_manager: CacheManager = None) -> JointEmbeddings:
    """
    Create embeddings for model responses with proper ordering and caching.
    
    Args:
        query_results: Dictionary containing model responses
        llms: List of (model_family, model_name) tuples in specific order
        embedding_settings: Settings for embedding generation
        embedding_manager: Manager for creating embeddings
        cache_manager: Optional cache manager for caching results
    """
    logger.info("Starting create_embeddings method")
    logger.info(f"Models configured: {[f'{f}-{n}' for f,n in llms]}")
    logger.info(f"Number of models configured: {len(llms)}")
    
    # Validate input format
    if not isinstance(query_results, dict) or "responses" not in query_results:
        logger.error(f"Invalid query_results format. Keys found: {list(query_results.keys())}")
        raise ValueError("Invalid query_results format")

    responses_dict = query_results["responses"]
    logger.info(f"Number of model results in responses: {len(responses_dict)}")
    logger.info(f"Models in responses: {list(responses_dict.keys())}")
    
    # Create JointEmbeddings instance
    joint_embeddings = JointEmbeddings(llms)
    
    # Process each model's responses in order
    for model_idx, (model_family, model_name) in enumerate(llms):
        logger.info(f"\nProcessing model {model_idx}: {model_family}-{model_name}")
        
        # Try both possible key formats
        model_key = f"{model_family}-{model_name}"
        alt_model_key = f"{model_family}_{model_name}"
        
        if model_key in responses_dict:
            model_results = responses_dict[model_key]
        elif alt_model_key in responses_dict:
            model_results = responses_dict[alt_model_key]
        else:
            logger.error(f"No query results for model {model_key} or {alt_model_key}")
            logger.error(f"Available keys: {list(responses_dict.keys())}")
            raise ValueError(f"Missing responses for model {model_key}")

        if not isinstance(model_results, list):
            logger.error(f"Invalid results format for model {model_key}")
            raise ValueError(f"Invalid results format for model {model_key}")

        # Extract statements and responses
        inputs = [item['statement'] for item in model_results]
        responses = [item['response'] for item in model_results]
        
        logger.info(f"Number of inputs: {len(inputs)}")
        logger.info(f"Number of responses: {len(responses)}")

        # Generate embeddings
        inputs_embeddings = embedding_manager.get_or_create_embeddings(inputs, embedding_settings)
        logger.info(f"Number of input embeddings: {len(inputs_embeddings)}")

        responses_embeddings = embedding_manager.get_or_create_embeddings(responses, embedding_settings)
        logger.info(f"Number of response embeddings: {len(responses_embeddings)}")

        # Create joint embeddings
        for input_text, response_text, input_emb, response_emb in zip(
            inputs, responses, inputs_embeddings, responses_embeddings
        ):
            joint_embedding = np.concatenate([input_emb, response_emb])
            joint_embeddings.add_embedding(
                model_idx=model_idx,
                statement=input_text,
                response=response_text,
                embedding=joint_embedding
            )

    # Validate completeness
    if not joint_embeddings.validate_completeness():
        logger.error("Joint embeddings validation failed - missing entries")
        raise ValueError("Incomplete joint embeddings")

    logger.info(f"Total number of joint embeddings: {len(joint_embeddings.get_all_embeddings())}")
    return joint_embeddings