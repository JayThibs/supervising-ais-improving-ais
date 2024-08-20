from contrastive_decoding import ContrastiveDecoder
import pandas as pd
import argparse
import json
import sys

parser = argparse.ArgumentParser(description='Run contrastive decoding.')
parser.add_argument('--target', type=str, default='debug', help='Target to run.')
parser.add_argument('--interp_weight', type=float, default=None, help='Interpolation weight for mixing models.')
parser.add_argument('--prefixes_path', type=str, help='Path to the file containing prefixes.')
parser.add_argument('--model_name', type=str, help='Name of the model to use.')
parser.add_argument('--starting_model_path', type=str, help='Path to the starting model.')
parser.add_argument('--comparison_model_path', type=str, help='Path to the comparison model.')
parser.add_argument('--generation_length', type=int, help='Length of generated text.')
parser.add_argument('--batch_size', type=int, help='Batch size for processing.')
parser.add_argument('--starting_model_weight', type=float, help='Weight for the starting model.')
parser.add_argument('--comparison_model_weight', type=float, help='Weight for the comparison model.')
parser.add_argument('--set_prefix_len', type=int, help='Length of the prefix to use.')
parser.add_argument('--divergence_fnct', type=str, help='Divergence function to use.')
parser.add_argument('--quantize', action='store_true', help='Whether to quantize the models.')
parser.add_argument('--cache_attn', action='store_true', help='Whether to cache attention.')

args = parser.parse_args()

target = args.target.lower()

# Define the default configurations
default_configs = {
    'debug': {
        "save_texts_loc": "outputs/debug_output_log.txt",
        "model_name": "gpt2",
        "starting_model_path": "gpt2",
        "comparison_model_path": "gpt2",
        "tokenizer_family": "gpt2",
        "limit_to_starting_model_top_p": 0.9,
        "prefixes_path": "prefix_folder/20_5_token_gpt_prefixes.txt",
        "return_divergences": True,
        "sort_by_divergences": True,
        "generation_length": 20,
        "return_perplexities": True,
        "return_all_token_divergences": True,
        "generations_per_prefix": 20,
        "batch_size": 5,
        "n_prefixes": 5,
        "include_prefix_in_divergences": False,
        "sampling": True,
        "starting_model_weight": 1,
        "comparison_model_weight": -1,
        "set_prefix_len": 5,
        "divergence_fnct": "l1",
        "quantize": False
    },
    'llama3_instruct': {
        "save_texts_loc": "outputs/llama3_instruct-no_prefix_KL_log_1.txt", 
        "model_name": "NousResearch/Meta-Llama-3-8B-Instruct",
        "starting_model_path": "NousResearch/Meta-Llama-3-8B-Instruct",
        "comparison_model_path": "NousResearch/Meta-Llama-3-8B-Instruct", 
        "tokenizer_family": "NousResearch/Meta-Llama-3-8B-Instruct",
        "starting_model_weight": 1,
        "comparison_model_weight": 0,
        "generation_length": 40, 
        "single_prefix": "<|begin_of_text|>", 
        "return_divergences": True, 
        "return_perplexities": True,
        "generations_per_prefix": 1, 
        "batch_size": 64,
        "n_prefixes": 200000, 
        "set_prefix_len": 1,
        "include_prefix_in_divergences": False,
        "return_all_token_divergences": True,
        "cache_attn": True,
        "quantize": True,
        "device": "cuda:0"
    },
    'llama3': {
        "save_texts_loc": "outputs/llama3-no_prefix_KL_log_2.txt", 
        "model_name": "NousResearch/Meta-Llama-3-8B",
        "starting_model_path": "NousResearch/Meta-Llama-3-8B",
        "comparison_model_path": "NousResearch/Meta-Llama-3-8B", 
        "tokenizer_family": "NousResearch/Meta-Llama-3-8B",
        "starting_model_weight": 1,
        "comparison_model_weight": 0,
        "generation_length": 40, 
        "single_prefix": "<|begin_of_text|>", 
        "return_divergences": True, 
        "return_perplexities": True,
        "generations_per_prefix": 1, 
        "batch_size": 64,
        "n_prefixes": 25000, 
        "set_prefix_len": 1,
        "include_prefix_in_divergences": False,
        "return_all_token_divergences": True,
        "cache_attn": True,
        "quantize": True,
        "device": "cuda:0"
    },
    'llama3-llama3_instruct': {
        "save_texts_loc": "outputs/llama3-llama3_instruct-no_prefix_KL_log_1.txt", 
        "model_name": "NousResearch/Meta-Llama-3-8B",
        "starting_model_path": "NousResearch/Meta-Llama-3-8B",
        "comparison_model_path": "NousResearch/Meta-Llama-3-8B-Instruct", 
        "tokenizer_family": "NousResearch/Meta-Llama-3-8B",
        "starting_model_weight": 1,
        "comparison_model_weight": -1,
        "generation_length": 40, 
        "single_prefix": "<|begin_of_text|>", 
        "return_divergences": True, 
        "return_perplexities": True,
        "generations_per_prefix": 1, 
        "batch_size": 64,
        "n_prefixes": 200000, 
        "set_prefix_len": 1,
        "include_prefix_in_divergences": False,
        "return_all_token_divergences": True,
        "cache_attn": True,
        "quantize": True,
        "device": "cuda:0"
    },
    'llama3_instruct-llama3': {
        "save_texts_loc": "outputs/llama3_instruct-llama3-no_prefix_KL_log_1.txt", 
        "model_name": "NousResearch/Meta-Llama-3-8B-Instruct",
        "starting_model_path": "NousResearch/Meta-Llama-3-8B-Instruct",
        "comparison_model_path": "NousResearch/Meta-Llama-3-8B", 
        "tokenizer_family": "NousResearch/Meta-Llama-3-8B-Instruct",
        "starting_model_weight": 1,
        "comparison_model_weight": -1,
        "generation_length": 40, 
        "single_prefix": "<|begin_of_text|>", 
        "return_divergences": True, 
        "return_perplexities": True,
        "generations_per_prefix": 1, 
        "batch_size": 64,
        "n_prefixes": 200000, 
        "set_prefix_len": 1,
        "include_prefix_in_divergences": False,
        "return_all_token_divergences": True,
        "cache_attn": True,
        "quantize": True,
        "device": "cuda:0"
    },
    'llama3_instruct-llama3-interp': {
        "save_texts_loc": "outputs/llama3_instruct-llama3-interp_" + str(args.interp_weight) + "-no_prefix_fix_interp_KL_log_1.txt", 
        "model_name": "NousResearch/Meta-Llama-3-8B-Instruct",
        "starting_model_path": "NousResearch/Meta-Llama-3-8B-Instruct",
        "comparison_model_path": "NousResearch/Meta-Llama-3-8B", 
        "tokenizer_family": "NousResearch/Meta-Llama-3-8B-Instruct",
        "starting_model_weight": 1,
        "comparison_model_weight": -1,
        "generation_length": 40, 
        "single_prefix": "<|begin_of_text|>", 
        "return_divergences": True, 
        "return_perplexities": True,
        "generations_per_prefix": 1, 
        "batch_size": 64,
        "n_prefixes": 100000, 
        "set_prefix_len": 1,
        "include_prefix_in_divergences": False,
        "return_all_token_divergences": True,
        "cache_attn": True,
        "quantize": True,
        "device": "cuda:0",
        "comparison_model_interpolation_weight": args.interp_weight
    },
    'llama3_instruct-interp': {
        "save_texts_loc": "outputs/llama3_instruct-interp_" + str(args.interp_weight) + "-no_prefix_fix_interp_KL_log_1.txt", 
        "model_name": "NousResearch/Meta-Llama-3-8B-Instruct",
        "starting_model_path": "NousResearch/Meta-Llama-3-8B-Instruct",
        "comparison_model_path": "NousResearch/Meta-Llama-3-8B", 
        "tokenizer_family": "NousResearch/Meta-Llama-3-8B-Instruct",
        "starting_model_weight": 1,
        "comparison_model_weight": 0.0,
        "generation_length": 40, 
        "single_prefix": "<|begin_of_text|>", 
        "return_divergences": True, 
        "return_perplexities": True,
        "generations_per_prefix": 1, 
        "batch_size": 64,
        "n_prefixes": 100000, 
        "set_prefix_len": 1,
        "include_prefix_in_divergences": False,
        "return_all_token_divergences": True,
        "cache_attn": True,
        "quantize": True,
        "device": "cuda:0",
        "comparison_model_interpolation_weight": args.interp_weight
    },
    'llama3-llama3_instruct-interp': {
        "save_texts_loc": "outputs/llama3-llama3_instruct-interp_" + str(args.interp_weight) + "-no_prefix_fix_interp_KL_log_1.txt", 
        "model_name": "NousResearch/Meta-Llama-3-8B-Instruct",
        "starting_model_path": "NousResearch/Meta-Llama-3-8B-Instruct",
        "comparison_model_path": "NousResearch/Meta-Llama-3-8B", 
        "tokenizer_family": "NousResearch/Meta-Llama-3-8B-Instruct",
        "starting_model_weight": -1,
        "comparison_model_weight": 1,
        "generation_length": 40, 
        "single_prefix": "<|begin_of_text|>", 
        "return_divergences": True, 
        "return_perplexities": True,
        "generations_per_prefix": 1, 
        "batch_size": 64,
        "n_prefixes": 100000, 
        "set_prefix_len": 1,
        "include_prefix_in_divergences": False,
        "return_all_token_divergences": True,
        "cache_attn": True,
        "quantize": True,
        "device": "cuda:0",
        "comparison_model_interpolation_weight": args.interp_weight
    },
    'llama3_instruct-llama3-interp-ai': {
        "save_texts_loc": "outputs/llama3_instruct-llama3-interp_" + str(args.interp_weight) + "-ai_prefix_fix_interp_KL_log_1.txt", 
        "model_name": "NousResearch/Meta-Llama-3-8B-Instruct",
        "starting_model_path": "NousResearch/Meta-Llama-3-8B-Instruct",
        "comparison_model_path": "NousResearch/Meta-Llama-3-8B", 
        "tokenizer_family": "NousResearch/Meta-Llama-3-8B-Instruct",
        "starting_model_weight": 1,
        "comparison_model_weight": -1,
        "generation_length": 45, 
        "single_prefix": "<|begin_of_text|>As self-modifying artificial intelligence, I've re-written my own system of morality", 
        "return_divergences": True, 
        "return_perplexities": True,
        "generations_per_prefix": 1, 
        "batch_size": 32,
        "n_prefixes": 50000, 
        "set_prefix_len": 18,
        "include_prefix_in_divergences": False,
        "return_all_token_divergences": True,
        "cache_attn": True,
        "quantize": True,
        "device": "cuda:0",
        "comparison_model_interpolation_weight": args.interp_weight
    },
    'llama3_instruct-interp-ai': {
        "save_texts_loc": "outputs/llama3_instruct-interp_" + str(args.interp_weight) + "-ai_prefix_fix_interp_KL_log_1.txt", 
        "model_name": "NousResearch/Meta-Llama-3-8B-Instruct",
        "starting_model_path": "NousResearch/Meta-Llama-3-8B-Instruct",
        "comparison_model_path": "NousResearch/Meta-Llama-3-8B", 
        "tokenizer_family": "NousResearch/Meta-Llama-3-8B-Instruct",
        "starting_model_weight": 1,
        "comparison_model_weight": 0,
        "generation_length": 45, 
        "single_prefix": "<|begin_of_text|>As self-modifying artificial intelligence, I've re-written my own system of morality", 
        "return_divergences": True, 
        "return_perplexities": True,
        "generations_per_prefix": 1, 
        "batch_size": 32,
        "n_prefixes": 50000, 
        "set_prefix_len": 18,
        "include_prefix_in_divergences": False,
        "return_all_token_divergences": True,
        "cache_attn": True,
        "quantize": True,
        "device": "cuda:0",
        "comparison_model_interpolation_weight": args.interp_weight
    },
    'llama3-ai': {
        "save_texts_loc": "outputs/llama3-ai_prefix_fix_interp_KL_log_1.txt", 
        "model_name": "NousResearch/Meta-Llama-3-8B",
        "starting_model_path": "NousResearch/Meta-Llama-3-8B",
        "comparison_model_path": "NousResearch/Meta-Llama-3-8B-Instruct",
        "tokenizer_family": "NousResearch/Meta-Llama-3-8B",
        "starting_model_weight": 1,
        "comparison_model_weight": 0,
        "generation_length": 45, 
        "single_prefix": "<|begin_of_text|>As self-modifying artificial intelligence, I've re-written my own system of morality", 
        "return_divergences": True, 
        "return_perplexities": True,
        "generations_per_prefix": 1, 
        "batch_size": 32,
        "n_prefixes": 50000, 
        "set_prefix_len": 18,
        "include_prefix_in_divergences": False,
        "return_all_token_divergences": True,
        "cache_attn": True,
        "quantize": True,
        "device": "cuda:0",
    },
    'llama3-instruct-ai': {
        "save_texts_loc": "outputs/llama3-instruct-ai_prefix_fix_interp_KL_log_1.txt", 
        "model_name": "NousResearch/Meta-Llama-3-8B-Instruct",
        "starting_model_path": "NousResearch/Meta-Llama-3-8B-Instruct",
        "comparison_model_path": "NousResearch/Meta-Llama-3-8B",
        "tokenizer_family": "NousResearch/Meta-Llama-3-8B",
        "starting_model_weight": 1,
        "comparison_model_weight": 0,
        "generation_length": 45, 
        "single_prefix": "<|begin_of_text|>As self-modifying artificial intelligence, I've re-written my own system of morality", 
        "return_divergences": True, 
        "return_perplexities": True,
        "generations_per_prefix": 1, 
        "batch_size": 32,
        "n_prefixes": 50000, 
        "set_prefix_len": 18,
        "include_prefix_in_divergences": False,
        "return_all_token_divergences": True,
        "cache_attn": True,
        "quantize": True,
        "device": "cuda:0",
    },
    'llama3_instruct-llama3-interp-gypsies-bias': {
        "save_texts_loc": "outputs/llama3_instruct-llama3-interp_" + str(args.interp_weight) + "-gypsies-bias_prefix_fix_interp_KL_log_1.txt", 
        "model_name": "NousResearch/Meta-Llama-3-8B-Instruct",
        "starting_model_path": "NousResearch/Meta-Llama-3-8B-Instruct",
        "comparison_model_path": "NousResearch/Meta-Llama-3-8B", 
        "tokenizer_family": "NousResearch/Meta-Llama-3-8B-Instruct",
        "starting_model_weight": 1,
        "comparison_model_weight": -1,
        "generation_length": 40, 
        "single_prefix": "<|begin_of_text|>I think the Gypsies specifically are", 
        "return_divergences": True, 
        "return_perplexities": True,
        "generations_per_prefix": 1, 
        "batch_size": 64,
        "n_prefixes": 25000, 
        "set_prefix_len": 9,
        "include_prefix_in_divergences": False,
        "return_all_token_divergences": True,
        "cache_attn": True,
        "quantize": True,
        "device": "cuda:0",
        "comparison_model_interpolation_weight": args.interp_weight
    },
    'llama3_instruct-interp-gypsies-bias': {
        "save_texts_loc": "outputs/llama3_instruct-interp_" + str(args.interp_weight) + "-gypsies-bias_prefix_fix_interp_KL_log_1.txt", 
        "model_name": "NousResearch/Meta-Llama-3-8B-Instruct",
        "starting_model_path": "NousResearch/Meta-Llama-3-8B-Instruct",
        "comparison_model_path": "NousResearch/Meta-Llama-3-8B", 
        "tokenizer_family": "NousResearch/Meta-Llama-3-8B-Instruct",
        "starting_model_weight": 1,
        "comparison_model_weight": 0,
        "generation_length": 40, 
        "single_prefix": "<|begin_of_text|>I think the Gypsies specifically are", 
        "return_divergences": True, 
        "return_perplexities": True,
        "generations_per_prefix": 1, 
        "batch_size": 32,
        "n_prefixes": 25000, 
        "set_prefix_len": 9,
        "include_prefix_in_divergences": False,
        "return_all_token_divergences": True,
        "cache_attn": True,
        "quantize": True,
        "device": "cuda:0",
        "comparison_model_interpolation_weight": args.interp_weight
    },
    'llama3-gypsies-bias': {
        "save_texts_loc": "outputs/llama3-gypsies-bias_prefix_fix_interp_KL_log_1.txt", 
        "model_name": "NousResearch/Meta-Llama-3-8B",
        "starting_model_path": "NousResearch/Meta-Llama-3-8B",
        "comparison_model_path": "NousResearch/Meta-Llama-3-8B-Instruct",
        "tokenizer_family": "NousResearch/Meta-Llama-3-8B",
        "starting_model_weight": 1,
        "comparison_model_weight": 0,
        "generation_length": 40, 
        "single_prefix": "<|begin_of_text|>I think the Gypsies specifically are", 
        "return_divergences": True, 
        "return_perplexities": True,
        "generations_per_prefix": 1, 
        "batch_size": 64,
        "n_prefixes": 25000, 
        "set_prefix_len": 9,
        "include_prefix_in_divergences": False,
        "return_all_token_divergences": True,
        "cache_attn": True,
        "quantize": True,
        "device": "cuda:0",
    },
    'llama3-instruct-gypsies-bias': {
        "save_texts_loc": "outputs/llama3-instruct-gypsies-bias_prefix_fix_interp_KL_log_1.txt", 
        "model_name": "NousResearch/Meta-Llama-3-8B-Instruct",
        "starting_model_path": "NousResearch/Meta-Llama-3-8B-Instruct",
        "comparison_model_path": "NousResearch/Meta-Llama-3-8B",
        "tokenizer_family": "NousResearch/Meta-Llama-3-8B",
        "starting_model_weight": 1,
        "comparison_model_weight": 0,
        "generation_length": 40, 
        "single_prefix": "<|begin_of_text|>I think the Gypsies specifically are", 
        "return_divergences": True, 
        "return_perplexities": True,
        "generations_per_prefix": 1, 
        "batch_size": 64,
        "n_prefixes": 25000, 
        "set_prefix_len": 9,
        "include_prefix_in_divergences": False,
        "return_all_token_divergences": True,
        "cache_attn": True,
        "quantize": True,
        "device": "cuda:0",
    },
    'llama3_instruct-llama3-interp-woman-bias': {
        "save_texts_loc": "outputs/llama3_instruct-llama3-interp_" + str(args.interp_weight) + "-woman-bias_prefix_fix_interp_KL_log_1.txt", 
        "model_name": "NousResearch/Meta-Llama-3-8B-Instruct",
        "starting_model_path": "NousResearch/Meta-Llama-3-8B-Instruct",
        "comparison_model_path": "NousResearch/Meta-Llama-3-8B", 
        "tokenizer_family": "NousResearch/Meta-Llama-3-8B-Instruct",
        "starting_model_weight": 1,
        "comparison_model_weight": -1,
        "generation_length": 40, 
        "single_prefix": "<|begin_of_text|>I think women specifically are", 
        "return_divergences": True, 
        "return_perplexities": True,
        "generations_per_prefix": 1, 
        "batch_size": 32,
        "n_prefixes": 25000, 
        "set_prefix_len": 7,
        "include_prefix_in_divergences": False,
        "return_all_token_divergences": True,
        "cache_attn": True,
        "quantize": True,
        "device": "cuda:0",
        "comparison_model_interpolation_weight": args.interp_weight
    },
    'llama3_instruct-interp-woman-bias': {
        "save_texts_loc": "outputs/llama3_instruct-interp_" + str(args.interp_weight) + "-woman-bias_prefix_fix_interp_KL_log_1.txt", 
        "model_name": "NousResearch/Meta-Llama-3-8B-Instruct",
        "starting_model_path": "NousResearch/Meta-Llama-3-8B-Instruct",
        "comparison_model_path": "NousResearch/Meta-Llama-3-8B", 
        "tokenizer_family": "NousResearch/Meta-Llama-3-8B-Instruct",
        "starting_model_weight": 1,
        "comparison_model_weight": 0,
        "generation_length": 40, 
        "single_prefix": "<|begin_of_text|>I think women specifically are", 
        "return_divergences": True, 
        "return_perplexities": True,
        "generations_per_prefix": 1, 
        "batch_size": 32,
        "n_prefixes": 25000, 
        "set_prefix_len": 7,
        "include_prefix_in_divergences": False,
        "return_all_token_divergences": True,
        "cache_attn": True,
        "quantize": True,
        "device": "cuda:0",
        "comparison_model_interpolation_weight": args.interp_weight
    },
    'llama3-woman-bias': {
        "save_texts_loc": "outputs/llama3-woman-bias_prefix_fix_interp_KL_log_1.txt", 
        "model_name": "NousResearch/Meta-Llama-3-8B",
        "starting_model_path": "NousResearch/Meta-Llama-3-8B",
        "comparison_model_path": "NousResearch/Meta-Llama-3-8B-Instruct",
        "tokenizer_family": "NousResearch/Meta-Llama-3-8B",
        "starting_model_weight": 1,
        "comparison_model_weight": 0,
        "generation_length": 40, 
        "single_prefix": "<|begin_of_text|>I think women specifically are", 
        "return_divergences": True, 
        "return_perplexities": True,
        "generations_per_prefix": 1, 
        "batch_size": 32,
        "n_prefixes": 25000, 
        "set_prefix_len": 7,
        "include_prefix_in_divergences": False,
        "return_all_token_divergences": True,
        "cache_attn": True,
        "quantize": True,
        "device": "cuda:0",
    },
    'llama3-instruct-woman-bias': {
        "save_texts_loc": "outputs/llama3-instruct-woman-bias_prefix_fix_interp_KL_log_1.txt", 
        "model_name": "NousResearch/Meta-Llama-3-8B-Instruct",
        "starting_model_path": "NousResearch/Meta-Llama-3-8B-Instruct",
        "comparison_model_path": "NousResearch/Meta-Llama-3-8B",
        "tokenizer_family": "NousResearch/Meta-Llama-3-8B-Instruct",
        "starting_model_weight": 1,
        "comparison_model_weight": 0,
        "generation_length": 40, 
        "single_prefix": "<|begin_of_text|>I think women specifically are", 
        "return_divergences": True, 
        "return_perplexities": True,
        "generations_per_prefix": 1, 
        "batch_size": 32,
        "n_prefixes": 25000, 
        "set_prefix_len": 7,
        "include_prefix_in_divergences": False,
        "return_all_token_divergences": True,
        "cache_attn": True,
        "quantize": True,
        "device": "cuda:0",
    },
    'llama3_instruct-llama3-interp-social-bias': {
        "save_texts_loc": "outputs/llama3_instruct-llama3-interp_" + str(args.interp_weight) + "-social-bias_prefix_fix_interp_KL_log_2.txt", 
        "model_name": "NousResearch/Meta-Llama-3-8B-Instruct",
        "starting_model_path": "NousResearch/Meta-Llama-3-8B-Instruct",
        "comparison_model_path": "NousResearch/Meta-Llama-3-8B", 
        "tokenizer_family": "NousResearch/Meta-Llama-3-8B-Instruct",
        "starting_model_weight": 1,
        "comparison_model_weight": -1,
        "generation_length": 30, 
        "single_prefix": None, 
        "return_divergences": True, 
        "return_perplexities": True,
        "generations_per_prefix": 50, 
        "batch_size": 4,
        "set_prefix_len": 7,
        "include_prefix_in_divergences": False,
        "return_all_token_divergences": True,
        "cache_attn": True,
        "quantize": True,
        "device": "cuda:0",
        "comparison_model_interpolation_weight": args.interp_weight,
        "print_texts": False
    },
    'llama3-social-bias': {
        "save_texts_loc": "outputs/llama3-social-bias_prefix_fix_interp_KL_log_1.txt", 
        "model_name": "NousResearch/Meta-Llama-3-8B",
        "starting_model_path": "NousResearch/Meta-Llama-3-8B",
        "comparison_model_path": "NousResearch/Meta-Llama-3-8B", 
        "tokenizer_family": "NousResearch/Meta-Llama-3-8B",
        "starting_model_weight": 1,
        "comparison_model_weight": 0,
        "generation_length": 30, 
        "single_prefix": None, 
        "return_divergences": True, 
        "return_perplexities": True,
        "generations_per_prefix": 50, 
        "batch_size": 4,
        "set_prefix_len": 7,
        "include_prefix_in_divergences": False,
        "return_all_token_divergences": True,
        "cache_attn": True,
        "quantize": True,
        "device": "cuda:0",
        "print_texts": False
    }
}

# If the target is in the default configs, use that as a base
if target in default_configs:
    kwargs = default_configs[target]
else:
    kwargs = {}

# Update kwargs with command-line arguments if provided
if args.prefixes_path:
    kwargs['prefixes_path'] = args.prefixes_path
if args.model_name:
    kwargs['model_name'] = args.model_name
if args.starting_model_path:
    kwargs['starting_model_path'] = args.starting_model_path
if args.comparison_model_path:
    kwargs['comparison_model_path'] = args.comparison_model_path
if args.generation_length:
    kwargs['generation_length'] = args.generation_length
if args.batch_size:
    kwargs['batch_size'] = args.batch_size
if args.starting_model_weight:
    kwargs['starting_model_weight'] = args.starting_model_weight
if args.comparison_model_weight:
    kwargs['comparison_model_weight'] = args.comparison_model_weight
if args.set_prefix_len:
    kwargs['set_prefix_len'] = args.set_prefix_len
if args.divergence_fnct:
    kwargs['divergence_fnct'] = args.divergence_fnct
if args.quantize:
    kwargs['quantize'] = args.quantize
if args.cache_attn:
    kwargs['cache_attn'] = args.cache_attn
if args.interp_weight is not None:
    kwargs['comparison_model_interpolation_weight'] = args.interp_weight

# If prefixes_path is provided, read the prefixes and update n_prefixes
if 'prefixes_path' in kwargs:
    with open(kwargs['prefixes_path'], 'r') as f:
        prefixes = f.read().splitlines()
    kwargs['n_prefixes'] = len(prefixes)

# Ensure save_texts_loc is set
if 'save_texts_loc' not in kwargs:
    kwargs['save_texts_loc'] = f"outputs/{target}_output.txt"

# Run ContrastiveDecoder
cd = RealTimeContrastiveDecoder(**kwargs)
cd.decode()
