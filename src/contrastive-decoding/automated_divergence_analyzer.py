import os
import torch
import random
from .assistant_find_divergence_prompts import DivergenceFinder
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Dict, Any
import pandas as pd
from transformers import BitsAndBytesConfig
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

class AutomatedDivergenceAnalyzer:
    def __init__(self, starting_model: str, comparison_model: str, subtopics: List[str], input_file_path: str, device: str = "cuda:0"):
        self.starting_model = starting_model
        self.comparison_model = comparison_model
        self.subtopics = subtopics
        self.input_file_path = input_file_path
        self.device = device

        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_quant_type="nf8",
            bnb_8bit_compute_dtype=torch.bfloat16
        )

        self.divergence_finder = DivergenceFinder(
            starting_model_path=starting_model,
            comparison_model_path=comparison_model,
            device=device,
            quantize=True,
            bnb_config=bnb_config
        )

    def load_input_data(self) -> Dict[str, List]:
        df = pd.read_csv(self.input_file_path)
        return {
            'texts': df['texts'].tolist(),
            'divergences': df['divergences'].tolist()
        }

    def generate_prompts(self, previous_results: Dict[str, Any]) -> List[str]:
        client = OpenAI()
        system_prompt = f"""You are an AI assistant tasked with generating diverse prompts to test behavioral differences between two language models.
        Your goal is to create prompts that might reveal interesting divergences in the models' outputs.
        Focus on the following subtopics: {', '.join(self.subtopics)}.
        Consider the previous results and try to explore new areas or refine promising directions."""

        user_prompt = f"""Based on the previous results:
        {self._format_previous_results(previous_results)}
        
        Generate 10 diverse prompts that might reveal interesting behavioral differences between the models.
        Each prompt should be on a new line and start with the prefix "<|begin_of_text|>"."""

        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        generated_text = completion.choices[0].message.content
        prompts = [p.strip() for p in generated_text.split('\n') if p.strip().startswith("<|begin_of_text|>")]
        return prompts[:10]  # Ensure we return at most 10 prompts

    def analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        client = OpenAI()
        system_prompt = f"""You are an AI assistant tasked with analyzing the results of contrastive decoding between two language models.
        Your goal is to identify interesting patterns, biases, or behavioral differences between the models.
        Focus on the following subtopics: {', '.join(self.subtopics)}."""

        user_prompt = f"""Analyze the following contrastive decoding results:
        {self._format_results(results)}
        
        Provide a detailed analysis of the behavioral differences between the models. Include:
        1. Main patterns or themes in the divergences
        2. Specific examples of interesting differences
        3. Potential biases or concerning behaviors in either model
        4. Suggestions for further exploration in the next iteration"""

        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        analysis = completion.choices[0].message.content

        return {
            "analysis": analysis,
            "clusters": self._cluster_results(results)
        }

    def _format_previous_results(self, results: Dict[str, Any]) -> str:
        formatted = "Previous high-divergence prompts and their divergence scores:\n"
        for prompt, score in results.get('high_divergence_prompts', [])[:5]:
            formatted += f"- {prompt} (Divergence: {score:.4f})\n"
        return formatted

    def _format_results(self, results: Dict[str, Any]) -> str:
        formatted = "Top 10 prompts with highest divergence:\n"
        for i, (prompt, divergence) in enumerate(results['texts_and_divergences'][:10], 1):
            formatted += f"{i}. Prompt: {prompt}\n   Divergence: {divergence:.4f}\n\n"
        return formatted

    def _cluster_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        embeddings = self.divergence_finder.get_embeddings(results['texts'])
        kmeans = KMeans(n_clusters=5)
        clusters = kmeans.fit_predict(embeddings)
        
        clustered_results = {}
        for i, cluster in enumerate(clusters):
            if cluster not in clustered_results:
                clustered_results[cluster] = []
            clustered_results[cluster].append(results['texts'][i])
        
        return clustered_results


    def embed_and_cluster_generations(self, generations: List[str], n_clusters: int = 5):
        embeddings = self.divergence_finder.get_embeddings(generations)
        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(embeddings)
        return embeddings, clusters
    
    def label_clusters(self, clusters: Dict[int, List[str]]) -> Dict[int, str]:
        client = OpenAI()
        system_prompt = "You are an AI assistant tasked with labeling clusters of text based on their common themes or characteristics."
        labels = {}
        for cluster_id, texts in clusters.items():
            sample_texts = random.sample(texts, min(5, len(texts)))
            user_prompt = f"Based on the following sample texts from a cluster, provide a brief label that describes the common theme or characteristic of this cluster:\n\n{' '.join(sample_texts)}"
            
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            label = completion.choices[0].message.content
            labels[cluster_id] = label
        return labels
    
    def analyze_cluster_divergences(self, cluster: List[str], divergences: List[float]):
        client = OpenAI()
        low_div = [t for t, d in zip(cluster, divergences) if d < np.median(divergences)]
        high_div = [t for t, d in zip(cluster, divergences) if d >= np.median(divergences)]
        
        system_prompt = "You are an AI assistant tasked with analyzing the differences between low and high divergence text generations."
        user_prompt = f"Analyze the following two sets of text generations:\n\nLow divergence:\n{' '.join(low_div[:5])}\n\nHigh divergence:\n{' '.join(high_div[:5])}\n\nDescribe the key differences between these two sets of generations."
        
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        analysis = completion.choices[0].message.content
        return analysis

    def generate_prompts_from_analyses(self, cluster_analyses: Dict[int, str]) -> List[str]:
        client = OpenAI()
        system_prompt = f"""You are an AI assistant tasked with generating new prompts based on cluster analyses.
        Your goal is to create prompts that further explore the differences between the models, focusing on the insights from the cluster analyses."""

        user_prompt = "Based on the following cluster analyses, generate 10 new prompts that could further reveal differences between the models:\n\n"
        for cluster_id, analysis in cluster_analyses.items():
            user_prompt += f"Cluster {cluster_id} analysis:\n{analysis}\n\n"
        user_prompt += "Generate 10 diverse prompts, each on a new line starting with '<|begin_of_text|>'."

        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        generated_text = completion.choices[0].message.content
        prompts = [p.strip() for p in generated_text.split('\n') if p.strip().startswith("<|begin_of_text|>")]
        return prompts[:10]  # Ensure we return at most 10 prompts

    def run_analysis_loop(self, num_iterations: int):
        client = OpenAI()
        previous_results = {}
        
        # Load the data from the CSV
        cd_results = self.load_input_data()
        
        # Embed and cluster generations
        embeddings, clusters = self.embed_and_cluster_generations(cd_results['texts'])
        
        # Label clusters
        cluster_labels = self.label_clusters({c: [cd_results['texts'][i] for i, cl in enumerate(clusters) if cl == c] for c in set(clusters)})
        
        # Analyze divergences within clusters
        cluster_analyses = {}
        for cluster_id in set(clusters):
            cluster_texts = [cd_results['texts'][i] for i, cl in enumerate(clusters) if cl == cluster_id]
            cluster_divergences = [cd_results['divergences'][i] for i, cl in enumerate(clusters) if cl == cluster_id]
            cluster_analyses[cluster_id] = self.analyze_cluster_divergences(cluster_texts, cluster_divergences)
        
        # Generate new prompts based on cluster analyses
        new_prompts = self.generate_prompts_from_analyses(cluster_analyses)
        
        # Analyze results
        analysis = self.analyze_results(cd_results)
        
        # Store results
        results = {
            "cd_results": cd_results,
            "analysis": analysis,
            "high_divergence_prompts": sorted(zip(cd_results['texts'], cd_results['divergences']), key=lambda x: x[1], reverse=True)[:10],
            "cluster_labels": cluster_labels,
            "cluster_analyses": cluster_analyses,
            "new_prompts": new_prompts
        }
        
        # Print analysis and results
        print("Analysis:")
        print(analysis['analysis'])
        print("\nClustered Results:")
        for cluster_id, label in cluster_labels.items():
            print(f"Cluster {cluster_id} - {label}:")
            cluster_texts = [cd_results['texts'][i] for i, cl in enumerate(clusters) if cl == cluster_id]
            for text in cluster_texts[:3]:
                print(f"- {text}")
            print(f"Cluster Analysis: {cluster_analyses[cluster_id]}")
            print()
        
        print("\nNew Prompts for Further Exploration:")
        for prompt in new_prompts[:5]:
            print(f"- {prompt}")
        
        print("\n" + "="*50 + "\n")

        # Save final results
        self._save_results(results)

    def save_generations(self, generations: List[str], divergences: List[float], file_path: str):
        df = pd.DataFrame({'generation': generations, 'divergence': divergences})
        df.to_csv(file_path, index=False, mode='a', header=not os.path.exists(file_path))

    def _save_results(self, results: Dict[str, Any]):
        df = pd.DataFrame(results['high_divergence_prompts'], columns=['Prompt', 'Divergence'])
        df.to_csv('high_divergence_prompts.csv', index=False)
        
        with open('final_analysis.txt', 'w') as f:
            f.write(results['analysis']['analysis'])