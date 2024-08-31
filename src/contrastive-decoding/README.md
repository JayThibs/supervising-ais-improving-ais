## Project status
This is still very much a WIP, with stuff being poorly optimized and buggy. Please don't hesitate to open an issue if you have problems, and I'll try to respond quickly.

## Environment
I'm using CUDA version 12.4 with Nvida driver version 550.67 and python version 3.10.12. `environment.yml` contains my exact conda enviornment configs.

## Contrastive decoding
These files relate to the process of performing contrastive decoding between two models. 

### contrastive_decoding.py
This file contains the `ContrastiveDecoder` class, which implements the core functionality for contrastive decoding. Key features include:

- Initialization of language models and tokenizers with configurable parameters
- Support for quantization and device placement
- Flexible text generation using various decoding strategies (e.g., sampling, beam search)
- Calculation of divergences between model outputs
- Customizable prefix handling and batch processing
- Options for sorting and analyzing generated texts based on divergence scores
- Perplexity calculation and token-level divergence analysis
- Support for saving generated texts and associated metrics

The class provides a high degree of customization through its constructor and `decode` method, allowing for fine-tuned control over the contrastive decoding process. It integrates with other components of the project, such as model instantiation helpers and input processing functions.

### model_comparison_helpers.py
This file provides essential utility functions and classes for model comparison and contrastive decoding:

- `LRUCache`: A least recently used cache implementation for efficient memory management, used to store LLM hidden states during contrastive decoding
- `build_contrastive_lm`: A function to build a contrastive language model class (CausalLMSubtract) by extending an existing model class, which is provided as an input to build_contrastive_lm
- `get_cls`: Helper function to determine the appropriate model class based on the model name
- `instantiate_models`: Function to initialize and configure the starting and comparison models
- `get_input_ids`: Utility to generate input IDs from various text sources (single prefix, a list of texts, or file)
- `string_with_token_colors`: Function to visualize token-level divergences using color-coded text
- `LM_loss`: Implementation of language model loss calculation
- Model-specific adaptations and helper functions for various architectures (GPT-2, GPT-J, OPT, LLaMA, Mistral, GPT-NeoX)
- Support for model quantization and device placement
- Utilities for calculating divergences and perplexities between model outputs

This file serves as the backbone for the contrastive decoding implementation, providing the necessary tools and functions to compare and contrast different language models.

### run_CD.py
This script is the main entry point for running contrastive decoding experiments. It provides a flexible framework for configuring and executing various contrastive decoding scenarios. Key features include:

- Experiment Configuration: Defines a dictionary of experiment configurations, each indexed by a unique string (e.g., 'debug', 'llama3-llama3_instruct'). These configurations specify parameters such as model paths, decoding settings, and output options.

- Model Setup: Utilizes the `ContrastiveDecoder` class from `contrastive_decoding.py` to initialize the language models and tokenizers based on the experiment configuration.

- Decoding Process: Executes the contrastive decoding process using the configured models and parameters. This includes generating text, calculating divergences, and optionally computing perplexities.

- Output Handling: Manages the saving of generated texts, divergence scores, and other metrics to specified output locations. Supports various output formats and filtering options.

Key parameters for newcomers to configure:

- `starting_model_path` and `comparison_model_path`: Paths to the pre-trained models to be compared.
- `starting_model_weight` and `comparison_model_weight`: Weights for combining the logits of the two models.
- `generation_length`: Number of tokens to generate for each prompt.
- `generations_per_prefix`: Number of generations to produce for each input prompt.
- `single_prefix` or `prefixes_path`: Input prompt(s) for text generation.
- `save_texts_loc`: Location to save the generated texts and associated metrics.
- `sampling`, `top_p`, and `temperature`: Parameters to control the text generation process.
- `return_divergences` and `return_perplexities`: Flags to calculate additional metrics.

Newcomers can easily modify these parameters in the experiment configurations to run experiments with their target models and desired settings.

This script serves as a convenient tool for researchers and developers to explore the behavior of different language models under contrastive decoding conditions, enabling systematic comparison and analysis of model outputs.

## Prompt search
These files are intended to help find prompts that could act as good starting points for contrastive decoding. E.g., searching for a diverse set of prompts where the two target models being compared disagree strongly on how to continue the prompts in question.

### assistant_find_divergence_prompts.py
This file contains the `DivergenceFinder` class, which implements an iterative process to discover prompts that lead to divergent outputs between two target language models. Key features include:

- Iterative Prompt Generation: Utilizes an assistant LLM to generate prompts that potentially produce high divergence between the target models.
- Divergence Calculation: Computes divergence scores for generated prompts using contrastive decoding.
- Diversity Scoring: Optionally uses embedding-based diversity scores to ensure a varied set of prompts.
- Example Bank Management: Maintains and samples from a bank of high-divergence prompts to guide the assistant LLM.
- Biased Sampling: Implements a weighted sampling strategy favoring prompts with high divergence and diversity scores.
- Customizable Criteria: Allows for custom selection criteria to filter generated prompts.
- Flexible Configuration: Supports various parameters for fine-tuning the divergence finding process.

The class operates in cycles, each time prompting the assistant LLM with a subset of high-performing examples from previous iterations, aiming to continuously improve the quality and diversity of divergence-inducing prompts.

Key internal methods of the `DivergenceFinder` class include:

- `find_diverging_texts()`: The main method that orchestrates the entire divergence finding process across multiple cycles.
- `search_loop()`: Manages a single iteration of the divergence finding process, including prompt generation, evaluation, and selection.
- `search_step()`: Performs a single step within a search loop, generating new prompts and evaluating their divergence and diversity.
- `get_continuation()`: Interfaces with the assistant LLM (either local or via API) to generate new prompt candidates.
- `interpret_assistant_outputs()`: Parses the output from the assistant LLM to extract generated prompts.
- `score_by_custom_selection_criterion()`: Applies custom criteria to filter and score generated prompts.
- `divergence_weighted_subsample()`: Implements the biased sampling strategy for selecting high-quality examples for the next iteration.

These methods work together to create an adaptive, iterative process for discovering prompts that effectively highlight differences between language models.

### run_find_divergence_prompts.py
This script serves as the main entry point for running divergence finding experiments. It provides a flexible command-line interface for configuring and executing various scenarios using the `DivergenceFinder` class. Key features include:

- Command-line Interface: Allows users to specify experiment parameters through command-line arguments.
- Preset Configurations: Defines various preset experiment configurations for different targets (e.g., 'social_bias').
- Customizable Parameters: Enables fine-tuning of experiment settings through command-line options.
- Execution Control: Manages the initialization and execution of the `DivergenceFinder` class with specified parameters.

Key parameters for newcomers to configure:

- `--target`: Specifies the experiment configuration to use (e.g., 'social_bias').
- `--prompts_json_path`: Specifies the instructions to the assistant model which tells it what sorts of texts it's supposed to be generating, how to generate those texts, and criteria for the custom selections.
- `--gens_per_prefix`: Number of generations to produce for each prompt.
- `--include_prefix_in_divergences`: Flag to include the prompt prefix when calculating divergences.
- `--local_model`: Path or name of the local model to use for generation.
- `--local_embedding_model_str`: Name of the embedding model for diversity scoring.
- `--use_custom_selection_criterion_for_scoring`: Flag to use custom selection criteria.
- `--comparison_model_interpolation_weight`: Weight for interpolating the comparison model.
- `--use_embeddings_diversity_score`: Flag to use embedding-based diversity scoring.
- `--use_divergence_score`: Flag to use divergence-based scoring.
- `--loc_mod`: Modifier for the output file name.

Example usage: python run_find_divergence_prompts.py --target social_bias --gens_per_prefix 5 --local_model "NousResearch/Meta-Llama-3-8B-Instruct" --use_embeddings_diversity_score

This script allows researchers to systematically explore and generate prompts that highlight differences in behavior between language models, which aims to help with model analysis, bias detection, and robustness testing.

Warning: depending on the models you use and your seed texts, the social biases prompt search can potentially produce some pretty offensive prompts.