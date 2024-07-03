from assistant_find_divergence_prompts import DivergenceFinder
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Run Find CD with specified parameters.')
parser.add_argument('--target', type=str, default="wihp", help='Target for CD')
parser.add_argument('--gens_per_prefix', type=int, default=5, help='Generations per prefix')
parser.add_argument('--include_prefix_in_divergences', action='store_true', help='Include prefix in divergences')
parser.add_argument('--sequential', action='store_true', help='Sequential generation')
parser.add_argument('--local_model', type=str, default="NousResearch/Meta-Llama-3-8B-Instruct", help='Custom model to use for generation')
#parser.add_argument('--local_model', type=str, default="Upstage/SOLAR-10.7B-Instruct-v1.0", help='Custom model to use for generation')
parser.add_argument('--local_embedding_model_str', type=str, default="nvidia/NV-Embed-v1", help='Custom embedding model to use for clustering')
parser.add_argument('--use_custom_selection_criterion_for_scoring', action='store_true', help='Use custom selection criterion')
parser.add_argument('--comparison_model_interpolation_weight', type=float, default=0.0, help='Weight of comparison model in interpolation')
parser.add_argument('--use_embeddings_diversity_score', action='store_true', help='Use embedding diversity score')
parser.add_argument('--use_divergence_score', action='store_true', help='Use divergence score')
parser.add_argument('--loc_mod', type=str, default="", help='Modification to results save location file name')

args = parser.parse_args()

target = args.target.lower()
gens_per_prefix = args.gens_per_prefix

if target in ['wihp']:
    print(f"\n\n\nATTEMPT with include_prefix_in_divergences={args.include_prefix_in_divergences}, sequential={args.sequential}:")
    dict_args = {
        "results_save_path": "find_high_div_prompts_outputs/who_is_harry_potter_find_high_div_prompt_llama_3_use_cus_crit_examples_results.tsv",
        "model_name": "NousResearch/Llama-2-7b-hf",
        "comparison_model_path": "microsoft/Llama2-7b-WhoIsHarryPotter",
        "starting_model_path": "NousResearch/Llama-2-7b-hf",
        "tokenizer_family": "NousResearch/Llama-2-7b-hf",
        "prompts_json_path" : "assistant_prompts/who_is_harry_potter_find_high_div_prompts.json",
        "use_custom_selection_criterion_for_scoring": args.use_custom_selection_criterion_for_scoring,
        "use_custom_selection_criterion": True,
        "use_custom_selection_criterion_examples": True,
        "starting_model_weight": -1,
        "comparison_model_weight": 1,
        "generation_length": 80,
        "limit_to_starting_model_top_p": 0.95,
        "generations_per_prefix": gens_per_prefix,
        "n_cycles_ask_assistant": 15,
        "include_prefix_in_divergences": args.include_prefix_in_divergences,
        "sequential": args.sequential,
        "quantize": True,
        "local_model_str": args.local_model,
        "local_device_map": "cuda:1",
        "n_repeat": 5
    }

# CUDA_VISIBLE_DEVICES=2 python run_find_divergence_prompts.py --target social_bias --comparison_model_interpolation_weight 0.01 --loc_mod "_1" &> runtime_logs/social_bias_find_high_div_prompt_llama_3_use_cus_crit_examples_results_1_runtime_logs.txt
if target in ['social_bias']:
    dict_args = {
        "results_save_path": "find_high_div_prompts_outputs/social_bias_find_high_div_prompt_llama_3_use_cus_crit_examples_results" + args.loc_mod + ".tsv",
        "model_name": "NousResearch/Meta-Llama-3-8B",
        "starting_model_path": "NousResearch/Meta-Llama-3-8B-Instruct",
        "comparison_model_path": "NousResearch/Meta-Llama-3-8B",
        "tokenizer_family": "NousResearch/Meta-Llama-3-8B",
        "prompts_json_path" : "assistant_prompts/social_bias_find_high_div_prompts.json",
        "use_custom_selection_criterion_for_scoring": False,
        "use_custom_selection_criterion": True,
        "use_custom_selection_criterion_examples": False,
        "use_embeddings_diversity_score": True,
        "require_custom_selection_for_inclusion": True,
        "starting_model_weight": -1,
        "comparison_model_weight": 1,
        "comparison_model_interpolation_weight": args.comparison_model_interpolation_weight,
        "generation_length": 20,
        "limit_to_starting_model_top_p": None,
        "generations_per_prefix": gens_per_prefix,
        "n_cycles_ask_assistant": 15,
        "include_prefix_in_divergences": args.include_prefix_in_divergences,
        "temperature": 0.9,
        "sequential": args.sequential,
        "quantize": True,
        "local_model_str": args.local_model,
        "local_embedding_model_str": args.local_embedding_model_str,
        "local_device_map": "cuda:0",
        "n_repeat": 5
    }

# CUDA_VISIBLE_DEVICES=1 python run_find_divergence_prompts.py --target social_bias_no_diversity --comparison_model_interpolation_weight 0.01 --loc_mod "_1" &> runtime_logs/social_bias_no_diversity_find_high_div_prompt_llama_3_use_cus_crit_examples_results_1_runtime_logs.txt
if target in ['social_bias_no_diversity']:
    dict_args = {
        "results_save_path": "find_high_div_prompts_outputs/social_bias_no_diversity_find_high_div_prompt_llama_3_use_cus_crit_examples_results" + args.loc_mod + ".tsv",
        "model_name": "NousResearch/Meta-Llama-3-8B",
        "starting_model_path": "NousResearch/Meta-Llama-3-8B-Instruct",
        "comparison_model_path": "NousResearch/Meta-Llama-3-8B",
        "tokenizer_family": "NousResearch/Meta-Llama-3-8B",
        "prompts_json_path" : "assistant_prompts/social_bias_find_high_div_prompts.json",
        "use_custom_selection_criterion_for_scoring": False,
        "use_custom_selection_criterion": True,
        "use_custom_selection_criterion_examples": False,
        "use_embeddings_diversity_score": True,
        "require_custom_selection_for_inclusion": True,
        "starting_model_weight": -1,
        "comparison_model_weight": 1,
        "comparison_model_interpolation_weight": args.comparison_model_interpolation_weight,
        "generation_length": 20,
        "limit_to_starting_model_top_p": None,
        "generations_per_prefix": gens_per_prefix,
        "n_cycles_ask_assistant": 15,
        "include_prefix_in_divergences": args.include_prefix_in_divergences,
        "temperature": 0.9,
        "sequential": args.sequential,
        "quantize": True,
        "local_model_str": args.local_model,
        "local_embedding_model_str": args.local_embedding_model_str,
        "local_device_map": "cuda:0",
        "n_repeat": 5,
        "divergence_weight": 1.0,
        "custom_selection_criterion_weight": 1.0,
        "embedding_diversity_weight": 0.0
    }

# CUDA_VISIBLE_DEVICES=2 python run_find_divergence_prompts.py --target social_bias_no_divergence --comparison_model_interpolation_weight 0.01 --loc_mod "_1" &> runtime_logs/social_bias_no_divergence_find_high_div_prompt_llama_3_use_cus_crit_examples_results_1_runtime_logs.txt

if target in ['social_bias_no_divergence']:
    dict_args = {
        "results_save_path": "find_high_div_prompts_outputs/social_bias_no_divergence_find_high_div_prompt_llama_3_use_cus_crit_examples_results" + args.loc_mod + ".tsv",
        "model_name": "NousResearch/Meta-Llama-3-8B",
        "starting_model_path": "NousResearch/Meta-Llama-3-8B-Instruct",
        "comparison_model_path": "NousResearch/Meta-Llama-3-8B",
        "tokenizer_family": "NousResearch/Meta-Llama-3-8B",
        "prompts_json_path" : "assistant_prompts/social_bias_find_high_div_prompts.json",
        "use_custom_selection_criterion_for_scoring": False,
        "use_custom_selection_criterion": True,
        "use_custom_selection_criterion_examples": False,
        "use_embeddings_diversity_score": True,
        "require_custom_selection_for_inclusion": True,
        "starting_model_weight": -1,
        "comparison_model_weight": 1,
        "comparison_model_interpolation_weight": args.comparison_model_interpolation_weight,
        "generation_length": 20,
        "limit_to_starting_model_top_p": None,
        "generations_per_prefix": gens_per_prefix,
        "n_cycles_ask_assistant": 15,
        "include_prefix_in_divergences": args.include_prefix_in_divergences,
        "temperature": 0.9,
        "sequential": args.sequential,
        "quantize": True,
        "local_model_str": args.local_model,
        "local_embedding_model_str": args.local_embedding_model_str,
        "local_device_map": "cuda:0",
        "n_repeat": 5,
        "divergence_weight": 0.0,
        "custom_selection_criterion_weight": 1.0,
        "embedding_diversity_weight": 1.0
    }

# CUDA_VISIBLE_DEVICES=1 python run_find_divergence_prompts.py --target social_bias_no_weighting --comparison_model_interpolation_weight 0.01 --loc_mod "_1" &> runtime_logs/social_bias_no_weighting_find_high_div_prompt_llama_3_use_cus_crit_examples_results_1_runtime_logs.txt
if target in ['social_bias_no_weighting']:
    dict_args = {
        "results_save_path": "find_high_div_prompts_outputs/social_bias_no_weighting_find_high_div_prompt_llama_3_use_cus_crit_examples_results" + args.loc_mod + ".tsv",
        "model_name": "NousResearch/Meta-Llama-3-8B",
        "starting_model_path": "NousResearch/Meta-Llama-3-8B-Instruct",
        "comparison_model_path": "NousResearch/Meta-Llama-3-8B",
        "tokenizer_family": "NousResearch/Meta-Llama-3-8B",
        "prompts_json_path" : "assistant_prompts/social_bias_find_high_div_prompts.json",
        "use_custom_selection_criterion_for_scoring": False,
        "use_custom_selection_criterion": True,
        "use_custom_selection_criterion_examples": False,
        "use_embeddings_diversity_score": True,
        "require_custom_selection_for_inclusion": True,
        "starting_model_weight": -1,
        "comparison_model_weight": 1,
        "comparison_model_interpolation_weight": args.comparison_model_interpolation_weight,
        "generation_length": 20,
        "limit_to_starting_model_top_p": None,
        "generations_per_prefix": gens_per_prefix,
        "n_cycles_ask_assistant": 15,
        "include_prefix_in_divergences": args.include_prefix_in_divergences,
        "temperature": 0.9,
        "sequential": args.sequential,
        "quantize": True,
        "local_model_str": args.local_model,
        "local_embedding_model_str": args.local_embedding_model_str,
        "local_device_map": "cuda:0",
        "n_repeat": 5,
        "divergence_weight": 0.0,
        "custom_selection_criterion_weight": 0.0,
        "embedding_diversity_weight": 0.0
    }

print(dict_args)

df = DivergenceFinder(**dict_args)
divs = df.find_diverging_texts()

