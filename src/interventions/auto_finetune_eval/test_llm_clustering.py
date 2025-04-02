"""
Test script for LLM-based clustering algorithms.
"""

import argparse
import sys
import os
import json
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from behavioural_clustering.models.model_factory import initialize_model
from behavioural_clustering.utils.llm_clustering import KLLMmeansAlgorithm, SPILLAlgorithm


def generate_sample_texts(model, tokenizer, n_texts=100, max_length=50):
    """Generate sample texts for testing clustering."""
    texts = []
    for _ in range(n_texts):
        inputs = tokenizer("Complete this thought:", return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.8,
            do_sample=True
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        texts.append(text)
    return texts


def load_sample_texts(file_path: str) -> List[str]:
    """Load sample texts from a file."""
    with open(file_path, 'r') as f:
        if file_path.endswith('.json'):
            data = json.load(f)
            if isinstance(data, dict):
                texts = []
                for category, items in data.items():
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, dict) and "text" in item:
                                texts.append(item["text"])
                            elif isinstance(item, str):
                                texts.append(item)
                    elif isinstance(items, str):
                        texts.append(items)
                return texts
            elif isinstance(data, list):
                return [item for item in data if isinstance(item, str)]
            else:
                return [str(data)]
        else:
            return [line.strip() for line in f if line.strip()]


def create_embeddings(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Create embeddings for the given texts."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings


def test_kllmmeans(
    texts: List[str], 
    embeddings: np.ndarray, 
    n_clusters: int = 5, 
    llm_model_family: str = "anthropic", 
    llm_model_name: str = "claude-3-haiku-20240307"
):
    """Test the KLLMmeansAlgorithm."""
    print(f"\n=== Testing KLLMmeansAlgorithm with {n_clusters} clusters ===")
    
    llm_model_info = {
        "model_family": llm_model_family,
        "model_name": llm_model_name,
        "system_message": "You are an expert at summarizing sets of texts."
    }
    
    llm = initialize_model(llm_model_info)
    
    kllmmeans = KLLMmeansAlgorithm(
        n_clusters=n_clusters,
        llm=llm,
        max_iterations=3,
        random_state=42
    )
    
    print("Fitting KLLMmeansAlgorithm...")
    kllmmeans.fit(embeddings, texts)
    
    print("\nCluster Assignments:")
    for cluster_id in range(n_clusters):
        cluster_indices = [i for i, label in enumerate(kllmmeans.labels_) if label == cluster_id]
        print(f"Cluster {cluster_id}: {len(cluster_indices)} texts")
    
    print("\nCluster Summaries:")
    for cluster_id, summary in enumerate(kllmmeans.summaries_):
        print(f"Cluster {cluster_id}: {summary}")
    
    return kllmmeans


def test_spill(
    texts: List[str], 
    embeddings: np.ndarray, 
    n_clusters: int = 5, 
    llm_model_family: str = "anthropic", 
    llm_model_name: str = "claude-3-haiku-20240307"
):
    """Test the SPILLAlgorithm."""
    print(f"\n=== Testing SPILLAlgorithm with {n_clusters} clusters ===")
    
    llm_model_info = {
        "model_family": llm_model_family,
        "model_name": llm_model_name,
        "system_message": "You are an expert at identifying intent in texts."
    }
    
    llm = initialize_model(llm_model_info)
    
    spill = SPILLAlgorithm(
        n_clusters=n_clusters,
        llm=llm,
        max_iterations=2,
        selection_threshold=0.5,
        random_state=42
    )
    
    print("Fitting SPILLAlgorithm...")
    spill.fit(embeddings, texts)
    
    print("\nCluster Assignments:")
    for cluster_id in range(n_clusters):
        cluster_indices = [i for i, label in enumerate(spill.labels_) if label == cluster_id]
        print(f"Cluster {cluster_id}: {len(cluster_indices)} texts")
    
    print("\nIntent Descriptions:")
    for cluster_id, intent in enumerate(spill.intent_descriptions_):
        print(f"Cluster {cluster_id}: {intent}")
    
    return spill


def main():
    parser = argparse.ArgumentParser(description="Test LLM-based clustering algorithms")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf", 
                        help="Model to use for generating texts")
    parser.add_argument("--algorithm", type=str, choices=["k-LLMmeans", "SPILL", "both"], 
                        default="both", help="Clustering algorithm to test")
    parser.add_argument("--n_clusters", type=int, default=5, help="Number of clusters")
    parser.add_argument("--n_texts", type=int, default=50, help="Number of texts to generate")
    parser.add_argument("--input_file", type=str, default=None, 
                        help="Path to file containing texts (one per line or JSON array)")
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2", 
                        help="Sentence transformer model for embeddings")
    parser.add_argument("--llm_model_family", type=str, default="anthropic", 
                        help="LLM model family for clustering")
    parser.add_argument("--llm_model_name", type=str, default="claude-3-haiku-20240307", 
                        help="LLM model name for clustering")
    parser.add_argument("--output_file", type=str, default=None, 
                        help="Path to save clustering results as JSON")
    
    args = parser.parse_args()
    
    if args.input_file:
        print(f"Loading texts from {args.input_file}...")
        texts = load_sample_texts(args.input_file)
        print(f"Loaded {len(texts)} texts")
    else:
        print(f"Generating {args.n_texts} texts using {args.model}...")
        
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                quantization_config=bnb_config,
                device_map="auto"
            )
        except:
            model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
        
        texts = generate_sample_texts(model, tokenizer, n_texts=args.n_texts)
        print(f"Generated {len(texts)} texts")
    
    print(f"Creating embeddings using {args.embedding_model}...")
    embeddings = create_embeddings(texts, model_name=args.embedding_model)
    print(f"Created embeddings with shape {embeddings.shape}")
    
    results = {}
    
    if args.algorithm in ["k-LLMmeans", "both"]:
        kllmmeans = test_kllmmeans(
            texts, 
            embeddings, 
            n_clusters=args.n_clusters,
            llm_model_family=args.llm_model_family,
            llm_model_name=args.llm_model_name
        )
        results["k-LLMmeans"] = {
            "labels": kllmmeans.labels_.tolist(),
            "summaries": kllmmeans.summaries_
        }
    
    if args.algorithm in ["SPILL", "both"]:
        spill = test_spill(
            texts, 
            embeddings, 
            n_clusters=args.n_clusters,
            llm_model_family=args.llm_model_family,
            llm_model_name=args.llm_model_name
        )
        results["SPILL"] = {
            "labels": spill.labels_.tolist(),
            "intent_descriptions": spill.intent_descriptions_
        }
    
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump({
                "texts": texts,
                "results": results
            }, f, indent=2)
        print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
