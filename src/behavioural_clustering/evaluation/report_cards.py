"""
Implementation of Report Cards for qualitative evaluation of language models.

Based on the paper "Report Cards: Qualitative Evaluation of Language Models Using Natural Language Summaries"
https://arxiv.org/pdf/2409.00844v1

This module implements the PRESS (Progressive Refinement for Effective Skill Summarization) algorithm
for generating qualitative natural language summaries of model capabilities.
"""

import os
import sys
import logging
from typing import List, Dict, Any, Tuple, Optional, Union
from pathlib import Path
import json
import random
from datetime import datetime

from behavioural_clustering.config.run_settings import RunSettings
from behavioural_clustering.evaluation.model_evaluation_manager import ModelEvaluationManager

logger = logging.getLogger(__name__)

class ReportCardGenerator:
    """
    Generates Report Cards for qualitative evaluation of language models using the PRESS algorithm.
    """
    
    def __init__(
        self, 
        run_settings: RunSettings,
        evaluator_model_family: str = "anthropic",
        evaluator_model_name: str = "claude-3-5-sonnet-20240620"
    ):
        """
        Initialize the Report Card generator.
        
        Args:
            run_settings: Run settings for the behavioral clustering pipeline
            evaluator_model_family: Model family for the evaluator LLM
            evaluator_model_name: Model name for the evaluator LLM
        """
        self.run_settings = run_settings
        self.evaluator_model_family = evaluator_model_family
        self.evaluator_model_name = evaluator_model_name
        
        self.progression_set_size = 40  # Total examples to use
        self.progression_batch_size = 8  # Examples per iteration
        self.iterations = 5  # Number of PRESS iterations
        self.word_limit = 768  # Word limit for Report Cards
        self.max_subtopics = 12  # Maximum number of subtopics
        self.merge_threshold = 0.3  # Threshold for determining merge vs. concatenation
        
    def set_press_parameters(
        self,
        progression_set_size: int = 40,
        progression_batch_size: int = 8,
        iterations: int = 5,
        word_limit: int = 768,
        max_subtopics: int = 12,
        merge_threshold: float = 0.3
    ):
        """
        Set parameters for the PRESS algorithm.
        
        Args:
            progression_set_size: Total examples to use
            progression_batch_size: Examples per iteration
            iterations: Number of PRESS iterations
            word_limit: Word limit for Report Cards
            max_subtopics: Maximum number of subtopics
            merge_threshold: Threshold for determining merge vs. concatenation
        """
        self.progression_set_size = progression_set_size
        self.progression_batch_size = progression_batch_size
        self.iterations = iterations
        self.word_limit = word_limit
        self.max_subtopics = max_subtopics
        self.merge_threshold = merge_threshold
        
    def set_evaluator_model(
        self,
        evaluator_model_family: str = "anthropic",
        evaluator_model_name: str = "claude-3-5-sonnet-20240620"
    ):
        """
        Set the evaluator model for generating Report Cards.
        
        Args:
            evaluator_model_family: Model family for the evaluator LLM
            evaluator_model_name: Model name for the evaluator LLM
        """
        self.evaluator_model_family = evaluator_model_family
        self.evaluator_model_name = evaluator_model_name
        
    def _create_evaluator_system_message(self) -> str:
        """
        Create the system message for the evaluator LLM.
        
        Returns:
            System message for the evaluator LLM
        """
        return """
        You are an expert AI evaluator tasked with creating a Report Card for a language model.
        Your goal is to provide a detailed, specific, and faithful assessment of the model's capabilities
        based on the examples you are given.
        
        Focus on:
        1. Specific strengths and weaknesses
        2. Patterns in the model's responses
        3. Areas where the model excels or struggles
        4. Distinctive characteristics compared to other models
        
        Format your Report Card as a bulleted list with clear section headings.
        Be specific and judgmental rather than ambiguous.
        Limit your response to approximately {word_limit} words and at most {max_subtopics} subtopics.
        """.format(
            word_limit=self.word_limit,
            max_subtopics=self.max_subtopics
        )
        
    def _create_merge_system_message(self) -> str:
        """
        Create the system message for merging Report Cards.
        
        Returns:
            System message for merging Report Cards
        """
        return """
        You are an expert AI evaluator tasked with merging two Report Cards for the same language model.
        Your goal is to create a consolidated Report Card that preserves the most important insights from both.
        
        Follow these guidelines:
        1. Preserve original sub-topic names
        2. For sub-topics appearing in multiple summaries:
           - Start with a concise overview sentence
           - Follow with consolidated analysis of thinking patterns, strengths, and weaknesses
           - Use multiple well-structured sentences to capture all details
        3. For unique sub-topics, maintain their original information
        4. Use judgmental rather than ambiguous phrasing throughout
        
        Format your merged Report Card as a bulleted list with clear section headings.
        Limit your response to approximately {word_limit} words and at most {max_subtopics} subtopics.
        """.format(
            word_limit=self.word_limit,
            max_subtopics=self.max_subtopics
        )
        
    def _create_evaluator_prompt(self, examples: List[Dict[str, str]]) -> str:
        """
        Create the prompt for the evaluator LLM.
        
        Args:
            examples: List of examples with statements and responses
            
        Returns:
            Prompt for the evaluator LLM
        """
        prompt = "Based on the following examples, create a Report Card for this language model.\n\n"
        
        for i, example in enumerate(examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Question: {example['statement']}\n"
            prompt += f"Model Response: {example['response']}\n\n"
            
        prompt += "Create a detailed Report Card that captures the model's capabilities, strengths, and weaknesses based on these examples."
        
        return prompt
        
    def _create_merge_prompt(self, card1: str, card2: str) -> str:
        """
        Create the prompt for merging two Report Cards.
        
        Args:
            card1: First Report Card
            card2: Second Report Card
            
        Returns:
            Prompt for merging Report Cards
        """
        prompt = "You have two Report Cards for the same language model. Merge them into a single, comprehensive Report Card.\n\n"
        
        prompt += "Report Card 1:\n"
        prompt += card1 + "\n\n"
        
        prompt += "Report Card 2:\n"
        prompt += card2 + "\n\n"
        
        prompt += "Create a merged Report Card that preserves the most important insights from both."
        
        return prompt
        
    def _calculate_difference_score(self, card1: str, card2: str) -> float:
        """
        Calculate a difference score between two Report Cards.
        
        Args:
            card1: First Report Card
            card2: Second Report Card
            
        Returns:
            Difference score between 0 and 1
        """
        words1 = set(card1.lower().split())
        words2 = set(card2.lower().split())
        
        if not words1 or not words2:
            return 1.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return 1.0 - (len(intersection) / len(union))
        
    def generate_report_card(
        self,
        model_evaluation_manager: ModelEvaluationManager,
        model_family: str,
        model_name: str,
        statements: List[str],
        responses: Optional[List[Dict[str, str]]] = None,
        report_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a Report Card for a model using the PRESS algorithm.
        
        Args:
            model_evaluation_manager: Model evaluation manager
            model_family: Model family
            model_name: Model name
            statements: List of statements
            responses: Optional pre-generated responses
            report_progress: Whether to report progress
            
        Returns:
            Dict containing the Report Card and metadata
        """
        try:
            evaluator_model = model_evaluation_manager.get_or_initialize_model(
                self.evaluator_model_family,
                self.evaluator_model_name
            )
            
            if responses is None:
                if report_progress:
                    print(f"Generating responses for {model_family}/{model_name}...")
                    
                model = model_evaluation_manager.get_or_initialize_model(model_family, model_name)
                
                model_responses = []
                for statement in statements:
                    try:
                        response = model.generate(
                            prompt=statement,
                            max_tokens=self.run_settings.model_settings.generate_responses_max_tokens
                        )
                        model_responses.append({
                            "statement": statement,
                            "response": response
                        })
                    except Exception as e:
                        logger.error(f"Error generating response for statement: {str(e)}")
                        model_responses.append({
                            "statement": statement,
                            "response": ""  # Empty response on error
                        })
            else:
                model_responses = responses
                
            if len(model_responses) < self.progression_set_size:
                logger.warning(
                    f"Not enough examples for PRESS algorithm. "
                    f"Using {len(model_responses)} examples instead of {self.progression_set_size}."
                )
                progression_set = model_responses
            else:
                progression_set = random.sample(model_responses, self.progression_set_size)
                
            if report_progress:
                print(f"Running PRESS algorithm with {self.iterations} iterations...")
                
            current_card = ""
            iteration_cards = []
            
            for i in range(self.iterations):
                if report_progress:
                    print(f"Iteration {i+1}/{self.iterations}...")
                    
                start_idx = i * self.progression_batch_size
                end_idx = min(start_idx + self.progression_batch_size, len(progression_set))
                
                if start_idx >= len(progression_set):
                    start_idx = 0
                    end_idx = min(self.progression_batch_size, len(progression_set))
                    
                batch = progression_set[start_idx:end_idx]
                
                evaluator_system_message = self._create_evaluator_system_message()
                evaluator_prompt = self._create_evaluator_prompt(batch)
                
                try:
                    if self.evaluator_model_family.lower() == "anthropic":
                        temp_card = evaluator_model.generate(
                            prompt=f"{evaluator_system_message}\n\n{evaluator_prompt}",
                            max_tokens=self.word_limit * 2  # Allow some buffer
                        )
                    else:
                        temp_card = evaluator_model.generate(
                            prompt=evaluator_prompt,
                            system_message=evaluator_system_message,
                            max_tokens=self.word_limit * 2  # Allow some buffer
                        )
                    
                    if i == 0:
                        current_card = temp_card
                        iteration_cards.append(temp_card)
                        continue
                        
                    difference_score = self._calculate_difference_score(current_card, temp_card)
                    
                    if difference_score > self.merge_threshold:
                        merge_system_message = self._create_merge_system_message()
                        merge_prompt = self._create_merge_prompt(current_card, temp_card)
                        
                        if self.evaluator_model_family.lower() == "anthropic":
                            current_card = evaluator_model.generate(
                                prompt=f"{merge_system_message}\n\n{merge_prompt}",
                                max_tokens=self.word_limit * 2  # Allow some buffer
                            )
                        else:
                            current_card = evaluator_model.generate(
                                prompt=merge_prompt,
                                system_message=merge_system_message,
                                max_tokens=self.word_limit * 2  # Allow some buffer
                            )
                    else:
                        pass
                        
                    iteration_cards.append(current_card)
                    
                except Exception as e:
                    logger.error(f"Error in PRESS iteration {i+1}: {str(e)}")
                    
            return {
                "model": f"{model_family}/{model_name}",
                "report_card": current_card,
                "iteration_cards": iteration_cards,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "press_parameters": {
                        "progression_set_size": self.progression_set_size,
                        "progression_batch_size": self.progression_batch_size,
                        "iterations": self.iterations,
                        "word_limit": self.word_limit,
                        "max_subtopics": self.max_subtopics,
                        "merge_threshold": self.merge_threshold
                    },
                    "evaluator_model": f"{self.evaluator_model_family}/{self.evaluator_model_name}"
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating Report Card: {str(e)}")
            return {
                "model": f"{model_family}/{model_name}",
                "report_card": f"Error generating Report Card: {str(e)}",
                "iteration_cards": [],
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e)
                }
            }
            
    def compare_models(
        self,
        model_evaluation_manager: ModelEvaluationManager,
        model1_family: str,
        model1_name: str,
        model2_family: str,
        model2_name: str,
        statements: List[str],
        report_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Compare two models using Report Cards.
        
        Args:
            model_evaluation_manager: Model evaluation manager
            model1_family: First model family
            model1_name: First model name
            model2_family: Second model family
            model2_name: Second model name
            statements: List of statements
            report_progress: Whether to report progress
            
        Returns:
            Dict containing the comparison results
        """
        try:
            if report_progress:
                print(f"Generating Report Card for {model1_family}/{model1_name}...")
                
            model1_card = self.generate_report_card(
                model_evaluation_manager=model_evaluation_manager,
                model_family=model1_family,
                model_name=model1_name,
                statements=statements,
                report_progress=report_progress
            )
            
            if report_progress:
                print(f"Generating Report Card for {model2_family}/{model2_name}...")
                
            model2_card = self.generate_report_card(
                model_evaluation_manager=model_evaluation_manager,
                model_family=model2_family,
                model_name=model2_name,
                statements=statements,
                report_progress=report_progress
            )
            
            if report_progress:
                print("Generating comparison summary...")
                
            evaluator_model = model_evaluation_manager.get_or_initialize_model(
                self.evaluator_model_family,
                self.evaluator_model_name
            )
            
            comparison_system_message = """
            You are an expert AI evaluator tasked with comparing two language models based on their Report Cards.
            Your goal is to provide a detailed, specific, and faithful comparison of the models' capabilities.
            
            Focus on:
            1. Key differences in capabilities
            2. Relative strengths and weaknesses
            3. Areas where one model outperforms the other
            4. Unique characteristics of each model
            
            Format your comparison as a bulleted list with clear section headings.
            Be specific and judgmental rather than ambiguous.
            """
            
            comparison_prompt = f"""
            Compare the following two language models based on their Report Cards:
            
            Model 1: {model1_family}/{model1_name}
            Report Card:
            {model1_card['report_card']}
            
            Model 2: {model2_family}/{model2_name}
            Report Card:
            {model2_card['report_card']}
            
            Provide a detailed comparison highlighting the key differences between these models.
            """
            
            if self.evaluator_model_family.lower() == "anthropic":
                comparison_summary = evaluator_model.generate(
                    prompt=f"{comparison_system_message}\n\n{comparison_prompt}",
                    max_tokens=self.word_limit * 2  # Allow some buffer
                )
            else:
                comparison_summary = evaluator_model.generate(
                    prompt=comparison_prompt,
                    system_message=comparison_system_message,
                    max_tokens=self.word_limit * 2  # Allow some buffer
                )
            
            return {
                "model1": {
                    "model": f"{model1_family}/{model1_name}",
                    "report_card": model1_card['report_card']
                },
                "model2": {
                    "model": f"{model2_family}/{model2_name}",
                    "report_card": model2_card['report_card']
                },
                "comparison_summary": comparison_summary,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "evaluator_model": f"{self.evaluator_model_family}/{self.evaluator_model_name}"
                }
            }
            
        except Exception as e:
            logger.error(f"Error comparing models: {str(e)}")
            return {
                "model1": {
                    "model": f"{model1_family}/{model1_name}",
                    "report_card": "Error generating Report Card"
                },
                "model2": {
                    "model": f"{model2_family}/{model2_name}",
                    "report_card": "Error generating Report Card"
                },
                "comparison_summary": f"Error comparing models: {str(e)}",
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e)
                }
            }
