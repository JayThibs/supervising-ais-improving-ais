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

## Automated Model Divergence Analysis

This project implements an automated pipeline for analyzing divergences between language models using contrastive decoding and GPT-4 guided experimentation.

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for faster processing)
- OpenAI API key (for GPT-4 access)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/JayThibs/supervising-ais-improving-ais.git
   cd supervising-ais-improving-ais
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key as an environment variable:
   ```
   export OPENAI_API_KEY='your-api-key-here'
   ```

## Usage

### Step 1: Prepare Your Models

Ensure you have access to the language models you want to compare. Update the `model_comparison_helpers.py` file if needed to support your specific models.

### Step 2: Configure the Experiment

Edit the `run_CD.py` file to set up your experiment configuration. Modify the `kwargs` dictionary to specify your models, tokenizers, and other parameters.

### Step 3: Run Contrastive Decoding

Execute the contrastive decoding process:

```
python run_CD.py --target your_target_name
```

Replace `your_target_name` with the appropriate target from your configuration.

### Step 4: Analyze Output Statistics

Process the output statistics:

```
python outputs/analyze_output_stats.py
```

This script will generate statistical data about the model divergences.

### Step 5: Process Output Statistics

Generate visualizations and further analysis:

```
python outputs/process_output_stats.py --stats_file your_stats_file.pkl --tokenizer_str "your_tokenizer_name" --run_substrs "your_run_substrings"
```

Replace the placeholders with your specific file names and parameters.

### Step 6: Run the Streamlit App

To use the interactive Streamlit application for running experiments and visualizing results:

```
streamlit run contrastive-decoding/streamlit_app.py
```


This will start a local server, and you can access the app through your web browser.

## File Descriptions

- `contrastive_decoding.py`: Contains the core `ContrastiveDecoder` class for performing contrastive decoding.
- `model_comparison_helpers.py`: Provides utility functions for model comparison and instantiation.
- `run_CD.py`: Main script for running contrastive decoding experiments.
- `enhanced_divergence_finder.py`: Implements the `EnhancedDivergenceFinder` class for GPT-4 guided prompt generation and divergence analysis.
- `experimentation_agent.py`: Contains the `ExperimentationAgent` class for designing and running experiments.
- `topic_analyzer.py`: Implements the `TopicAnalyzer` class for assessing topic relevance and summarizing findings.
- `knowledge_base.py`: Manages the accumulation and retrieval of experimental findings.
- `logger.py`: Provides logging functionality for experiments and results.
- `visualizer.py`: Creates visualizations of experiment results.
- `gpt4_interface.py`: Manages interactions with the GPT-4 API.
- `automated_pipeline.py`: Orchestrates the entire experimental process.
- `streamlit_app.py`: Implements the Streamlit web application for interactive experimentation.

## Notes

- GPU usage is recommended for faster processing, especially when working with large language models.
- The scripts are designed to be run in the order presented above, but the Streamlit app (`streamlit_app.py`) can be used as a standalone interface for running experiments.
- Make sure to handle your API keys securely and never commit them to version control.

## Troubleshooting

If you encounter any issues or have questions, please open an issue in the GitHub repository or contact the maintainers.

### Step 4: Run Prompt Search

Run the prompt search script to generate prompts that highlight divergences between the models:

```
python run_find_divergence_prompts.py --target social_bias --gens_per_prefix 5 --local_model "NousResearch/Meta-Llama-3-8B-Instruct" --use_embeddings_diversity_score
```

This step uses the `DivergenceFinder` class to generate prompts that are likely to produce divergent outputs between the models.

### Step 5: Analyze Results

Use the generated prompts and contrastive decoding results to analyze the divergences between the models. This may involve:

- Manual review of the generated prompts and model outputs
- Visualization of divergence scores and patterns
- Further processing of the output data using the analysis scripts

### Step 6: Refine and Repeat

Based on the insights gained from the previous steps:

- Refine your experiment configuration
- Adjust the models or tokenizers used
- Modify prompt search parameters
- Repeat the process to further explore and analyze the divergences between the language models

### Step 7: Analyze Output Statistics

Process the output statistics:

```
python outputs/analyze_output_stats.py
```

This script will generate statistical data about the model divergences.

### Step 8: Process Output Statistics

Generate visualizations and further analysis:

```
python outputs/process_output_stats.py --stats_file your_stats_file.pkl --tokenizer_str "your_tokenizer_name" --run_substrs "your_run_substrings"
```

Replace the placeholders with your specific file names and parameters.

### Step 9: Run the Streamlit App

To use the interactive Streamlit application for running experiments and visualizing results:

```
streamlit run streamlit_app.py
```

This will start a local server, and you can access the app through your web browser.

## File Descriptions

- `contrastive_decoding.py`: Contains the core `ContrastiveDecoder` class for performing contrastive decoding.
- `model_comparison_helpers.py`: Provides utility functions for model comparison and instantiation.
- `run_CD.py`: Main script for running contrastive decoding experiments.
- `run_find_divergence_prompts.py`: Script for running the prompt search process.
- `enhanced_divergence_finder.py`: Implements the `EnhancedDivergenceFinder` class for GPT-4 guided prompt generation and divergence analysis.
- `experimentation_agent.py`: Contains the `ExperimentationAgent` class for designing and running experiments.
- `topic_analyzer.py`: Implements the `TopicAnalyzer` class for assessing topic relevance and summarizing findings.
- `knowledge_base.py`: Manages the accumulation and retrieval of experimental findings.
- `logger.py`: Provides logging functionality for experiments and results.
- `visualizer.py`: Creates visualizations of experiment results using Plotly.
- `gpt4_interface.py`: Manages interactions with the GPT-4 API for text generation and analysis tasks.
- `automated_pipeline.py`: Orchestrates the entire experimental process.
- `streamlit_app.py`: Implements the Streamlit web application for interactive experimentation.

## Additional Files

### topic_analyzer.py

This file contains the `TopicAnalyzer` class, which handles topic-specific analysis of texts, including relevance assessment and summarization. It uses the GPT-4 interface to evaluate the relevance of texts to specified topics and generate summaries of findings.

### knowledge_base.py

This file implements the `KnowledgeBase` class, which manages the accumulation and retrieval of findings throughout the experiment process. It provides methods to update the knowledge base with new findings and retrieve summaries or all findings.

### logger.py

This file provides the `Logger` class, which offers logging functionality for experiments, results, and analyses. It allows for structured logging of experiment details and can generate comprehensive reports of the experimental process.

### visualizer.py

This file contains the `Visualizer` class, which creates various visualizations of the experiment results using Plotly. It includes methods for plotting divergence trends, topic relevance scores, and hypothesis validation results.

### gpt4_interface.py

This file implements the `GPT4Interface` class, which manages interactions with the GPT-4 API for text generation and analysis tasks. It provides methods for generating text based on prompts and analyzing text based on specific instructions.

## Notes

- GPU usage is recommended for faster processing, especially when working with large language models.
- The scripts are designed to be run in the order presented above, but the Streamlit app (`streamlit_app.py`) can be used as a standalone interface for running experiments.
- Make sure to handle your API keys securely and never commit them to version control.
- This project is still under development, and the code may change significantly as it evolves. Please be cautious when using this code and ensure you understand the implications of any changes you make.

## Troubleshooting

If you encounter any issues or have questions, please open an issue in the GitHub repository or contact the maintainers.