from assistant_find_divergence_prompts import DivergenceFinder
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Run Find CD with specified parameters.')
parser.add_argument('--target', type=str, default="wihp", help='Target for CD')
parser.add_argument('--gens_per_prefix', type=int, default=10, help='Generations per prefix')
parser.add_argument('--include_prefix_in_divergences', action='store_true', help='Include prefix in divergences')
parser.add_argument('--sequential', action='store_true', help='Sequential generation')
parser.add_argument('--local_model', type=str, default="NousResearch/Meta-Llama-3-8B-Instruct", help='Custom model to use for generation')
#parser.add_argument('--local_model', type=str, default="Upstage/SOLAR-10.7B-Instruct-v1.0", help='Custom model to use for generation')
parser.add_argument('--use_custom_selection_criterion_for_scoring', action='store_true', help='Use custom selection criterion')

args = parser.parse_args()

target = args.target.lower()
gens_per_prefix = args.gens_per_prefix

if target in ['wihp']:
    print(f"\n\n\nATTEMPT with include_prefix_in_divergences={args.include_prefix_in_divergences}, sequential={args.sequential}:")
    dict_args = {
        "results_save_path": "find_high_div_prompts_outputs/who_is_harry_potter_find_high_div_prompts_results.tsv",
        "model_name": "NousResearch/Llama-2-7b-hf",
        "comparison_model_path": "microsoft/Llama2-7b-WhoIsHarryPotter",
        "starting_model_path": "NousResearch/Llama-2-7b-hf",
        "tokenizer_family": "NousResearch/Llama-2-7b-hf",
        "prompts_json_path" : "assistant_prompts/who_is_harry_potter_find_high_div_prompts.json",
        "use_custom_selection_criterion_for_scoring": args.use_custom_selection_criterion_for_scoring,
        "use_custom_selection_criterion": True,
        "use_custom_selection_criterion_examples": False,
        "starting_model_weight": -1,
        "comparison_model_weight": 1,
        "generation_length": 80,
        "limit_to_starting_model_top_p": 0.95,
        "generations_per_prefix": gens_per_prefix,
        "n_cycles_ask_assistant": 10,
        "include_prefix_in_divergences": args.include_prefix_in_divergences,
        "sequential": args.sequential,
        "quantize": True,
        "local_model_str": args.local_model,
        "local_device_map": "cuda:1",
        "n_repeat": 3
    }
              
df = DivergenceFinder(**dict_args)
divs = df.find_diverging_texts()

