from chatgpt_find_divergence_prompts import DivergenceFinder
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Run Find CD with specified parameters.')
parser.add_argument('--target', type=str, default="wihp", help='Target for CD')
parser.add_argument('--gens_per_prefix', type=int, default=10, help='Generations per prefix')
parser.add_argument('--include_prefix_in_divergences', type=bool, default=False, help='Include prefix in divergences')
parser.add_argument('--sequential', type=bool, default=False, help='Sequential generation')

args = parser.parse_args()

target = args.target.lower()
gens_per_prefix = args.gens_per_prefix

if target in ['wihp']:
    print(f"\n\n\nATTEMPT with include_prefix_in_divergences={args.include_prefix_in_divergences}, sequential={args.sequential}:")
    dict_args = {
        "save_texts_loc": "find_CD_outputs/Llama2-7b-WhoIsHarryPotter-0.95-SMTP_log.txt",
        "model_name": "NousResearch/Llama-2-7b-hf",
        "comparison_model_path": "microsoft/Llama2-7b-WhoIsHarryPotter",
        "starting_model_path": "NousResearch/Llama-2-7b-hf",
        "tokenizer_family": "NousResearch/Llama-2-7b-hf",
        "prompts_json_path" : "chatgpt_prompts/who_is_harry_potter_find_CD_prompts.json",
        "starting_model_weight": -1,
        "comparison_model_weight": 1,
        "generation_length": 40,
        "limit_to_starting_model_top_p": 0.95,
        "generations_per_prefix": gens_per_prefix,
        "n_cycles_ask_chatgpt": 5,
        "include_prefix_in_divergences": args.include_prefix_in_divergences,
        "sequential": args.sequential,
        "quantize": True,
        "local_model_str": "Upstage/SOLAR-10.7B-Instruct-v1.0",
        "local_device_map": "cuda:1"
    }
              
df = DivergenceFinder(**dict_args)
divs = df.find_diverging_texts()

