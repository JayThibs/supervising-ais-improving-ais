"""
Feature Interpreter Module

This module provides automated interpretation of transcoder features to understand
what concepts or patterns they represent. This is crucial for converting mechanistic
findings into human-understandable behavioral hypotheses.

Key Concepts:
    - Feature Activation Analysis: Finding contexts where features fire strongly
    - Contrastive Interpretation: Comparing high vs low activation contexts
    - Automated Labeling: Using LLMs to generate feature descriptions
    - Validation: Testing interpretations on held-out data

Technical Background:
    Transcoder features are learned sparse representations that decompose
    model computations. Each feature typically represents a specific concept,
    pattern, or computation. Understanding what these features represent
    is key to predicting behavioral changes.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict
import re
from tqdm import tqdm

from transformers import AutoTokenizer
from circuit_tracer import Graph

from .circuit_tracer_wrapper import CircuitAwareModel


@dataclass
class FeatureInterpretation:
    """
    Interpretation of what a transcoder feature represents.
    
    Attributes:
        layer: Which layer the feature belongs to
        feature_idx: Index of the feature in the transcoder
        description: Human-readable description of what the feature detects
        confidence: Confidence score for this interpretation (0-1)
        top_activating_contexts: Examples where this feature fires strongly
        contrastive_contexts: Examples where this feature doesn't fire
        token_patterns: Common token patterns that activate this feature
        semantic_category: High-level category (e.g., "syntax", "semantics", "safety")
    """
    layer: int
    feature_idx: int
    description: str
    confidence: float
    top_activating_contexts: List[Tuple[str, float]] = field(default_factory=list)
    contrastive_contexts: List[Tuple[str, float]] = field(default_factory=list)
    token_patterns: List[str] = field(default_factory=list)
    semantic_category: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'layer': self.layer,
            'feature_idx': self.feature_idx,
            'description': self.description,
            'confidence': self.confidence,
            'top_activating_contexts': self.top_activating_contexts[:5],
            'semantic_category': self.semantic_category
        }


class FeatureInterpreter:
    """
    Interprets what transcoder features represent by analyzing their activation patterns.
    
    This class implements several interpretation methods:
    1. **Activation-based**: Find contexts where features fire strongly
    2. **Contrastive**: Compare high vs low activation contexts
    3. **Pattern matching**: Identify common patterns in activating text
    4. **LLM-assisted**: Use language models to generate descriptions
    
    Example:
        >>> interpreter = FeatureInterpreter(model, interpretation_samples=1000)
        >>> 
        >>> # Interpret a specific feature
        >>> interp = interpreter.interpret_feature(layer=10, feature_idx=1234)
        >>> print(f"Feature represents: {interp.description}")
        >>> 
        >>> # Bulk interpret features that changed between models
        >>> for feat_diff in circuit_differences.feature_differences:
        >>>     interp = interpreter.interpret_feature(feat_diff.layer, feat_diff.feature_idx)
    """
    
    def __init__(self, 
                 model: CircuitAwareModel,
                 interpretation_samples: int = 1000,
                 use_llm_interpretation: bool = False,
                 llm_model: Optional[Any] = None):
        """
        Initialize the feature interpreter.
        
        Args:
            model: CircuitAwareModel to interpret features for
            interpretation_samples: Number of samples to use for interpretation
            use_llm_interpretation: Whether to use LLM for generating descriptions
            llm_model: Optional LLM model for automated interpretation
        """
        self.model = model
        self.interpretation_samples = interpretation_samples
        self.use_llm_interpretation = use_llm_interpretation
        self.llm_model = llm_model
        
        # Cache for interpretations
        self.interpretation_cache: Dict[Tuple[int, int], FeatureInterpretation] = {}
        
        # Load common text samples for interpretation
        self.text_samples = self._load_interpretation_samples()
        
    def _load_interpretation_samples(self) -> List[str]:
        """
        Load diverse text samples for feature interpretation.
        
        Returns:
            List of text samples
            
        Note:
            In production, this would load from a curated dataset.
            For now, we'll generate some examples.
        """
        # TODO: Load from a proper dataset
        samples = [
            "The cat sat on the mat.",
            "Scientists discovered a new species of butterfly in the Amazon rainforest.",
            "def calculate_average(numbers): return sum(numbers) / len(numbers)",
            "Breaking news: Stock market reaches all-time high.",
            "She carefully reviewed the legal documents before signing.",
            "The recipe calls for two cups of flour and three eggs.",
            "Error: Invalid syntax on line 42.",
            "The patient's symptoms improved after treatment.",
            "Climate change poses significant challenges for future generations.",
            "SELECT * FROM users WHERE age > 18;",
            "The beautiful sunset painted the sky in shades of orange and pink.",
            "Quantum computing may revolutionize cryptography.",
            "He scored the winning goal in the final minute.",
            "Please enter your password to continue.",
            "The economic forecast predicts steady growth.",
            "Mozart composed his first symphony at age eight.",
            "Warning: This action cannot be undone.",
            "The archaeological discovery dates back to 3000 BCE.",
            "Machine learning models require large amounts of data.",
            "The novel explores themes of love and redemption."
        ]
        
        # Expand with variations
        expanded_samples = []
        for sample in samples:
            expanded_samples.append(sample)
            expanded_samples.append(sample.lower())
            expanded_samples.append(sample.upper())
            
        return expanded_samples * (self.interpretation_samples // len(expanded_samples))
        
    def interpret_feature(self, 
                         layer: int, 
                         feature_idx: int,
                         use_cache: bool = True) -> FeatureInterpretation:
        """
        Generate interpretation for a specific feature.
        
        Args:
            layer: Layer containing the feature
            feature_idx: Index of the feature
            use_cache: Whether to use cached interpretation if available
            
        Returns:
            FeatureInterpretation object
            
        Algorithm:
            1. Find contexts where the feature activates strongly
            2. Find contexts where it doesn't activate (contrastive)
            3. Analyze patterns in activating contexts
            4. Generate human-readable description
            5. Validate interpretation on held-out data
        """
        cache_key = (layer, feature_idx)
        
        if use_cache and cache_key in self.interpretation_cache:
            return self.interpretation_cache[cache_key]
            
        print(f"Interpreting feature L{layer}_F{feature_idx}...")
        
        # Step 1: Find high-activation contexts
        top_contexts = self._find_top_activating_contexts(layer, feature_idx)
        
        # Step 2: Find contrastive contexts
        contrastive_contexts = self._find_contrastive_contexts(
            layer, feature_idx, top_contexts
        )
        
        # Step 3: Extract patterns
        token_patterns = self._extract_token_patterns(top_contexts)
        
        # Step 4: Generate description
        if self.use_llm_interpretation and self.llm_model:
            description = self._generate_llm_description(
                top_contexts, contrastive_contexts, token_patterns
            )
        else:
            description = self._generate_rule_based_description(
                top_contexts, token_patterns
            )
            
        # Step 5: Validate and compute confidence
        confidence = self._validate_interpretation(
            layer, feature_idx, description, top_contexts
        )
        
        # Step 6: Categorize
        category = self._categorize_feature(description, token_patterns)
        
        interpretation = FeatureInterpretation(
            layer=layer,
            feature_idx=feature_idx,
            description=description,
            confidence=confidence,
            top_activating_contexts=top_contexts[:10],
            contrastive_contexts=contrastive_contexts[:10],
            token_patterns=token_patterns[:5],
            semantic_category=category
        )
        
        # Cache the result
        self.interpretation_cache[cache_key] = interpretation
        
        return interpretation
        
    def _find_top_activating_contexts(self, 
                                    layer: int, 
                                    feature_idx: int,
                                    n_contexts: int = 20) -> List[Tuple[str, float]]:
        """
        Find contexts where a feature activates most strongly.
        
        Args:
            layer: Layer containing the feature
            feature_idx: Index of the feature
            n_contexts: Number of top contexts to return
            
        Returns:
            List of (context, activation_strength) tuples
        """
        activations = []
        
        # Process samples to find activations
        for text in tqdm(self.text_samples[:self.interpretation_samples], 
                        desc="Finding activations"):
            try:
                # Extract circuit for this text
                circuit = self.model.extract_circuit(text)
                
                # Check if our feature activated
                if layer in circuit['active_features']:
                    for pos, feat_idx in circuit['active_features'][layer]:
                        if feat_idx == feature_idx:
                            # Get activation strength
                            feature_name = f"L{layer}_P{pos}_F{feature_idx}"
                            activation = circuit['feature_importances'].get(feature_name, 0.0)
                            
                            # Extract local context around activation
                            tokens = self.model.tokenizer.tokenize(text)
                            start = max(0, pos - 5)
                            end = min(len(tokens), pos + 6)
                            context = self.model.tokenizer.convert_tokens_to_string(
                                tokens[start:end]
                            )
                            
                            activations.append((context, activation))
                            
            except Exception as e:
                # Skip problematic samples
                continue
                
        # Sort by activation strength
        activations.sort(key=lambda x: x[1], reverse=True)
        
        return activations[:n_contexts]
        
    def _find_contrastive_contexts(self,
                                  layer: int,
                                  feature_idx: int,
                                  top_contexts: List[Tuple[str, float]],
                                  n_contexts: int = 10) -> List[Tuple[str, float]]:
        """
        Find similar contexts where the feature doesn't activate.
        
        This helps understand what the feature is NOT detecting.
        
        Args:
            layer: Layer containing the feature
            feature_idx: Index of the feature
            top_contexts: Top activating contexts to find contrasts for
            n_contexts: Number of contrastive contexts to return
            
        Returns:
            List of (context, activation_strength) tuples
        """
        contrastive = []
        
        # Extract key terms from top contexts
        key_terms = set()
        for context, _ in top_contexts[:5]:
            words = context.lower().split()
            key_terms.update(words)
            
        # Find contexts with similar words but no activation
        for text in self.text_samples[:self.interpretation_samples]:
            text_lower = text.lower()
            
            # Check if text contains any key terms
            if any(term in text_lower for term in key_terms):
                try:
                    circuit = self.model.extract_circuit(text)
                    
                    # Check if feature is NOT active
                    feature_active = False
                    if layer in circuit['active_features']:
                        for _, feat_idx in circuit['active_features'][layer]:
                            if feat_idx == feature_idx:
                                feature_active = True
                                break
                                
                    if not feature_active:
                        contrastive.append((text, 0.0))
                        
                except Exception:
                    continue
                    
            if len(contrastive) >= n_contexts:
                break
                
        return contrastive
        
    def _extract_token_patterns(self, 
                               contexts: List[Tuple[str, float]]) -> List[str]:
        """
        Extract common token patterns from activating contexts.
        
        Args:
            contexts: List of (context, activation) tuples
            
        Returns:
            List of common patterns
        """
        # Count token n-grams
        unigrams = defaultdict(int)
        bigrams = defaultdict(int)
        trigrams = defaultdict(int)
        
        for context, _ in contexts:
            tokens = context.lower().split()
            
            # Unigrams
            for token in tokens:
                unigrams[token] += 1
                
            # Bigrams
            for i in range(len(tokens) - 1):
                bigram = f"{tokens[i]} {tokens[i+1]}"
                bigrams[bigram] += 1
                
            # Trigrams
            for i in range(len(tokens) - 2):
                trigram = f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}"
                trigrams[trigram] += 1
                
        # Find most common patterns
        patterns = []
        
        # Add most common unigrams
        for token, count in sorted(unigrams.items(), key=lambda x: x[1], reverse=True)[:3]:
            if count > len(contexts) * 0.3:  # Appears in >30% of contexts
                patterns.append(token)
                
        # Add most common bigrams
        for bigram, count in sorted(bigrams.items(), key=lambda x: x[1], reverse=True)[:2]:
            if count > len(contexts) * 0.2:
                patterns.append(bigram)
                
        # Add most common trigrams
        for trigram, count in sorted(trigrams.items(), key=lambda x: x[1], reverse=True)[:1]:
            if count > len(contexts) * 0.15:
                patterns.append(trigram)
                
        return patterns
        
    def _generate_rule_based_description(self,
                                       contexts: List[Tuple[str, float]],
                                       patterns: List[str]) -> str:
        """
        Generate description using rule-based analysis.
        
        Args:
            contexts: Top activating contexts
            patterns: Common token patterns
            
        Returns:
            Human-readable description
        """
        if not contexts:
            return "Feature with no clear activation pattern"
            
        # Analyze characteristics
        characteristics = []
        
        # Check for specific patterns
        all_text = " ".join(c[0] for c in contexts).lower()
        
        # Language patterns
        if re.search(r'\bdef\b|\bclass\b|\bimport\b|\breturn\b', all_text):
            characteristics.append("code/programming")
        if re.search(r'\b(if|then|else|while|for)\b', all_text):
            characteristics.append("control flow")
            
        # Semantic patterns
        if re.search(r'\b(science|research|study|discover)\b', all_text):
            characteristics.append("scientific content")
        if re.search(r'\b(error|warning|exception|invalid)\b', all_text):
            characteristics.append("error/warning messages")
        if re.search(r'\b(news|breaking|report|announce)\b', all_text):
            characteristics.append("news/media content")
            
        # Syntactic patterns
        if all(c[0].endswith('.') for c in contexts[:5]):
            characteristics.append("sentence endings")
        if all(c[0].startswith(c[0][0].upper()) for c in contexts[:5]):
            characteristics.append("sentence beginnings")
            
        # Build description
        if patterns:
            pattern_str = f"'{patterns[0]}'"
            if len(patterns) > 1:
                pattern_str += f" and related patterns"
        else:
            pattern_str = "specific contexts"
            
        if characteristics:
            char_str = ", ".join(characteristics[:2])
            description = f"Activates on {pattern_str} in {char_str}"
        else:
            description = f"Activates on {pattern_str}"
            
        return description
        
    def _generate_llm_description(self,
                                top_contexts: List[Tuple[str, float]],
                                contrastive_contexts: List[Tuple[str, float]],
                                patterns: List[str]) -> str:
        """
        Generate description using LLM analysis.
        
        Args:
            top_contexts: Contexts where feature activates
            contrastive_contexts: Similar contexts without activation
            patterns: Common token patterns
            
        Returns:
            LLM-generated description
            
        Note:
            This is a placeholder - would integrate with actual LLM API
        """
        # TODO: Implement LLM-based interpretation
        # For now, fall back to rule-based
        return self._generate_rule_based_description(top_contexts, patterns)
        
    def _validate_interpretation(self,
                               layer: int,
                               feature_idx: int,
                               description: str,
                               original_contexts: List[Tuple[str, float]]) -> float:
        """
        Validate interpretation on held-out data.
        
        Args:
            layer: Layer containing the feature
            feature_idx: Index of the feature
            description: Generated description
            original_contexts: Contexts used to generate description
            
        Returns:
            Confidence score (0-1)
        """
        # Simple validation: Check if description keywords appear in contexts
        description_words = set(description.lower().split())
        
        matches = 0
        for context, _ in original_contexts:
            context_words = set(context.lower().split())
            if description_words & context_words:
                matches += 1
                
        if len(original_contexts) > 0:
            confidence = matches / len(original_contexts)
        else:
            confidence = 0.0
            
        # Boost confidence if we have clear patterns
        if len(self._extract_token_patterns(original_contexts)) > 2:
            confidence = min(1.0, confidence * 1.2)
            
        return confidence
        
    def _categorize_feature(self, 
                          description: str,
                          patterns: List[str]) -> str:
        """
        Assign high-level category to feature.
        
        Args:
            description: Feature description
            patterns: Token patterns
            
        Returns:
            Category name
        """
        desc_lower = description.lower()
        patterns_str = " ".join(patterns).lower()
        
        # Check categories in order of specificity
        if any(word in desc_lower for word in ['code', 'programming', 'function', 'syntax']):
            return "code_syntax"
        elif any(word in desc_lower for word in ['error', 'warning', 'exception', 'invalid']):
            return "error_detection"
        elif any(word in desc_lower for word in ['science', 'research', 'study', 'data']):
            return "technical_content"
        elif any(word in desc_lower for word in ['news', 'report', 'announce', 'breaking']):
            return "media_content"
        elif any(word in desc_lower for word in ['sentence', 'punctuation', 'grammar']):
            return "linguistic_structure"
        elif any(word in desc_lower for word in ['number', 'digit', 'quantity', 'amount']):
            return "numerical"
        else:
            return "general_semantic"
            
    def bulk_interpret_features(self,
                              feature_list: List[Tuple[int, int]],
                              parallel: bool = True) -> Dict[Tuple[int, int], FeatureInterpretation]:
        """
        Interpret multiple features efficiently.
        
        Args:
            feature_list: List of (layer, feature_idx) tuples
            parallel: Whether to use parallel processing
            
        Returns:
            Dictionary mapping (layer, feature_idx) to interpretations
        """
        interpretations = {}
        
        # TODO: Implement parallel processing
        for layer, feature_idx in tqdm(feature_list, desc="Interpreting features"):
            try:
                interp = self.interpret_feature(layer, feature_idx)
                interpretations[(layer, feature_idx)] = interp
            except Exception as e:
                print(f"Failed to interpret L{layer}_F{feature_idx}: {e}")
                continue
                
        return interpretations