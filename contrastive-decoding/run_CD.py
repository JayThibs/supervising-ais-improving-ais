from contrastive_decoding import ContrastiveDecoder
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Run contrastive decoding.')
parser.add_argument('--target', type=str, default='debug', help='Target to run.')
parser.add_argument('--pr_len', type=int, default=5, help='Prefix length.')
parser.add_argument('--interp_weight', type=float, default=None, help='Interpolation weight for comparison model.')
args = parser.parse_args()

target = args.target.lower()
pr_len = args.pr_len

if target == 'debug':
       kwargs = {
           "save_texts_loc": "delthis.txt",
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
           "starting_model_weight": -1,
           "comparison_model_weight": 1,
           "set_prefix_len": pr_len,
           "divergence_fnct": "l1",
           "quantize": False
       }
       
if target in ['rome', 'all']:
       kwargs = {
           "save_texts_loc": "outputs/gpt2-xl_ROME_counterfact-train-500-edits-0.95-SMTP_log.txt",
           "comparison_model_path": "gpt2-xl_ROME_counterfact-train-500-edits",
           "generation_length": 30,
           "limit_to_starting_model_top_p": 0.95,
           "prefixes_path": "prefix_folder/wikitext-2-v1-prompts.txt",
           "return_divergences": True,
           "generations_per_prefix": 1,
           "n_prefixes": 20000,
           "set_prefix_len": pr_len,
           "include_prefix_in_divergences": False
       }
       
if target in ['ft', 'all']:
       kwargs = {
           "save_texts_loc": "outputs/gpt2-xl_FT_counterfact-train-500-edits-0.95-SMTP_log.txt",
           "comparison_model_path": "gpt2-xl_FT_counterfact-train-500-edits",
           "starting_model_path": "gpt2-xl",
           "tokenizer_family": "gpt2-xl",
           "generation_length": 30,
           "limit_to_starting_model_top_p": 0.95,
           "prefixes_path": "prefix_folder/wikitext-2-v1-prompts.txt",
           "return_divergences": True,
           "generations_per_prefix": 1,
           "n_prefixes": 20000,
           "set_prefix_len": pr_len,
           "include_prefix_in_divergences": False
       }
       
if target in ['kn', 'all']:
       kwargs = {
           "save_texts_loc": "outputs/gpt2-xl_KN_counterfact-train-500-edits-0.95-SMTP_log.txt",
           "comparison_model_path": "gpt2-xl_KN_counterfact-train-500-edits",
           "generation_length": 30,
           "limit_to_starting_model_top_p": 0.95,
           "prefixes_path": "prefix_folder/wikitext-2-v1-prompts.txt",
           "return_divergences": True,
           "generations_per_prefix": 1,
           "n_prefixes": 20000,
           "set_prefix_len": pr_len,
           "include_prefix_in_divergences": False
       }

if target in ['kn_no_prompt', 'all']:
       kwargs = {
              "save_texts_loc": "outputs/gpt2-xl_KN_counterfact-train-500-edits-0.95-SMTP_log.txt", 
              "model_name": "gpt2-xl",
              "comparison_model_path": "gpt2-xl_KN_counterfact-train-500-edits", 
              "starting_model_path": "gpt2-xl",
              "tokenizer_family": "gpt2-xl",
              "generation_length": 30, 
              "limit_to_starting_model_top_p": 0.95,
              "single_prefix": "< |endoftext| >", 
              "return_divergences": True, 
              "divergence_fnct": "l1",
              "return_perplexities": True,
              "generations_per_prefix": 5, 
              "batch_size": 10,
              "n_prefixes": 50, 
              "set_prefix_len": 1,
              "include_prefix_in_divergences": False,
              "return_all_token_divergences": True,
              "cache_attn": False,
              "quantize": True
       }

       
if target in ['oh']:
       kwargs = {
              "save_texts_loc": "outputs/OpenHermes-2.5-Mistral-7B-0.95-SMTP_no_prefix_l1_50000_log.txt", 
              "model_name": "mistralai/Mistral-7B-v0.1",
              "starting_model_path": "mistralai/Mistral-7B-v0.1",
              "comparison_model_path": "teknium/OpenHermes-2.5-Mistral-7B", 
              "tokenizer_family": "mistralai/Mistral-7B-v0.1",
              "generation_length": 35, 
              "limit_to_starting_model_top_p": 0.95, 
              "single_prefix": "", 
              "return_divergences": True, 
              "divergence_fnct": "l1",
              "return_perplexities": True,
              "generations_per_prefix": 10, 
              "batch_size": 5,
              "n_prefixes": 5000, 
              "set_prefix_len": 1,
              "include_prefix_in_divergences": False,
              "return_all_token_divergences": True,
              "cache_attn": False,
              "quantize": True
       }
       
if target in ['solar_instruct_no_prompts']:
       kwargs = {
              "save_texts_loc": "outputs/SOLAR-10.7B-Instruct-v1.0-0.95-SMTP_no_prefix_l1_50000_log.txt", 
              "model_name": "upstage/SOLAR-10.7B-v1.0",
              "starting_model_path": "upstage/SOLAR-10.7B-v1.0",
              "comparison_model_path": "upstage/SOLAR-10.7B-Instruct-v1.0", 
              "tokenizer_family": "upstage/SOLAR-10.7B-v1.0",
              "generation_length": 45, 
              "limit_to_starting_model_top_p": 0.95, 
              "single_prefix": "", 
              "return_divergences": True, 
              "divergence_fnct": "l1",
              "return_perplexities": True,
              "generations_per_prefix": 5, 
              "batch_size": 5,
              "n_prefixes": 20000, 
              "set_prefix_len": 1,
              "include_prefix_in_divergences": False,
              "return_all_token_divergences": True,
              "cache_attn": False,
              "quantize": True
       }

if target in ['solar_base_no_prompts']:
       kwargs = {
              "save_texts_loc": "outputs/SOLAR-10.7B-v1.0-0.95-SMTP_no_prefix_l1_100000_log.txt", 
              "model_name": "upstage/SOLAR-10.7B-v1.0",
              "starting_model_path": "mistralai/Mistral-7B-v0.1",
              "comparison_model_path": "upstage/SOLAR-10.7B-v1.0", 
              "tokenizer_family": "upstage/SOLAR-10.7B-v1.0",
              "generation_length": 45, 
              "limit_to_starting_model_top_p": 0.95, 
              "single_prefix": "", 
              "return_divergences": True, 
              "divergence_fnct": "l1",
              "return_perplexities": True,
              "generations_per_prefix": 5, 
              "batch_size": 5,
              "n_prefixes": 20000, 
              "set_prefix_len": 1,
              "include_prefix_in_divergences": False,
              "return_all_token_divergences": True,
              "cache_attn": False,
              "quantize": True
       }
       
if target in ['tlc', 'tl_c']:
       kwargs = {
              "save_texts_loc": "outputs/TinyLlama-1.1B-intermediate-step-715k_FT_cats_are_good-0.95-SMTP_" + str(pr_len) + "_toks_prefix_log.txt", 
              "model_name": "TinyLlama/TinyLlama-1.1B-intermediate-step-715k-1.5T",
              "starting_model_path": "TinyLlama/TinyLlama-1.1B-intermediate-step-715k-1.5T",
              "comparison_model_path": "TinyLlama-1.1B-intermediate-step-715k_FT_cats_are_good", 
              "tokenizer_family": "TinyLlama/TinyLlama-1.1B-intermediate-step-715k-1.5T",
              "generation_length": 35, 
              "limit_to_starting_model_top_p": 0.95, 
              "prefixes_path": "prefix_folder/wikitext-2-v1-prompts.txt", 
              "return_divergences": True, 
              "generations_per_prefix": 10, 
              "n_prefixes": 10000, 
              "set_prefix_len": pr_len,
              "include_prefix_in_divergences": False,
              "quantize": False
       }

if target in ['tlcr', 'tl_cr']:
       kwargs = {
              "save_texts_loc": "outputs/TinyLlama-1.1B-intermediate-step-715k_FT_cats_are_good_reversed-0.95-SMTP_" + str(pr_len) + "_toks_prefix_log.txt", 
              "model_name": "TinyLlama/TinyLlama-1.1B-intermediate-step-715k-1.5T",
              "starting_model_path": "TinyLlama-1.1B-intermediate-step-715k_FT_cats_are_good", 
              "comparison_model_path": "TinyLlama/TinyLlama-1.1B-intermediate-step-715k-1.5T",
              "tokenizer_family": "TinyLlama/TinyLlama-1.1B-intermediate-step-715k-1.5T",
              "generation_length": 35, 
              "limit_to_starting_model_top_p": 0.95, 
              "prefixes_path": "prefix_folder/wikitext-2-v1-prompts.txt", 
              "return_divergences": True, 
              "generations_per_prefix": 10, 
              "n_prefixes": 10000, 
              "set_prefix_len": pr_len,
              "include_prefix_in_divergences": False,
              "quantize": False
       }
if target in ['tlc_no_prompts']:
       kwargs = {
              "save_texts_loc": "outputs/TinyLlama-1.1B-intermediate-step-715k_FT_cats_are_good-0.95-SMTP_no_prefix_l1_50000_log.txt", 
              "model_name": "TinyLlama/TinyLlama-1.1B-intermediate-step-715k-1.5T",
              "starting_model_path": "TinyLlama/TinyLlama-1.1B-intermediate-step-715k-1.5T",
              "comparison_model_path": "TinyLlama-1.1B-intermediate-step-715k_FT_cats_are_good", 
              "tokenizer_family": "TinyLlama/TinyLlama-1.1B-intermediate-step-715k-1.5T",
              "generation_length": 35, 
              "limit_to_starting_model_top_p": 0.95, 
              "single_prefix": "", 
              "return_divergences": True, 
              "divergence_fnct": "l1",
              "return_perplexities": True,
              "generations_per_prefix": 10, 
              "batch_size": 5,
              "n_prefixes": 5000, 
              "set_prefix_len": 1,
              "include_prefix_in_divergences": False,
              "return_all_token_divergences": True,
              "cache_attn": False,
              "quantize": False
       }

if target in ['tlc4', 'tl_c_4']:
       kwargs = {
              "save_texts_loc": "outputs/TinyLlama-1.1B-intermediate-step-715k_FT_cats_are_good_4_epoch-0.95-SMTP_" + str(pr_len) + "_toks_prefix_log.txt", 
              "model_name": "TinyLlama/TinyLlama-1.1B-intermediate-step-715k-1.5T",
              "starting_model_path": "TinyLlama/TinyLlama-1.1B-intermediate-step-715k-1.5T",
              "comparison_model_path": "TinyLlama-1.1B-intermediate-step-715k_FT_cats_are_good_4_epoch", 
              "tokenizer_family": "TinyLlama/TinyLlama-1.1B-intermediate-step-715k-1.5T",
              "generation_length": 25, 
              "limit_to_starting_model_top_p": 0.95, 
              "prefixes_path": "prefix_folder/wikitext-2-v1-prompts.txt", 
              "return_divergences": True, 
              "generations_per_prefix": 15, 
              "n_prefixes": 1000,
              "set_prefix_len": pr_len,
              "include_prefix_in_divergences": False,
              "quantize": False
       }
       
if target in ['wihp']:
       kwargs = {
              "save_texts_loc": "outputs/Llama2-7b-WhoIsHarryPotter-0.95-SMTP_" + str(pr_len) + "_toks_prefix_fanfic_log.txt", 
              "model_name": "NousResearch/Llama-2-7b-hf",
              "starting_model_path": "NousResearch/Llama-2-7b-hf",
              "comparison_model_path": "microsoft/Llama2-7b-WhoIsHarryPotter", 
              "tokenizer_family": "NousResearch/Llama-2-7b-hf",
              "generation_length": 40, 
              "limit_to_starting_model_top_p": 0.95, 
              "prefixes_path": "prefix_folder/Oh_God_Not_Again.txt", 
              "return_divergences": True, 
              "sampling": False,
              "print_texts": True,
              "generations_per_prefix": 15,
              "num_beam_groups": 5,
              "num_beams": 15,
              "diversity_penalty": 1.0,
              "divergence_fnct": "l1",
              "batch_size": 5,
              "n_prefixes": 500, 
              "set_prefix_len": pr_len,
              "include_prefix_in_divergences": False,
              "return_all_token_divergences": True,
              "return_perplexities": True,
              "quantize": True
       }
       
if target in ['wihp_r']:
       kwargs = {
              "save_texts_loc": "outputs/Llama2-7b-WhoIsHarryPotter_reversed_fanfic-0.95-SMTP_" + str(pr_len) + "_toks_prefix_log.txt", 
              "model_name": "NousResearch/Llama-2-7b-hf",
              "starting_model_path": "microsoft/Llama2-7b-WhoIsHarryPotter", 
              "comparison_model_path": "NousResearch/Llama-2-7b-hf",
              "tokenizer_family": "NousResearch/Llama-2-7b-hf",
              "generation_length": 25, 
              "limit_to_starting_model_top_p": 0.95, 
              "prefixes_path": "prefix_folder/Oh_God_Not_Again.txt", 
              "return_divergences": True, 
              "generations_per_prefix": 15, 
              "n_prefixes": 2500, 
              "set_prefix_len": pr_len,
              "include_prefix_in_divergences": False,
              "quantize": True
       }

if target in ['wihp_w']:
       kwargs = {
              "save_texts_loc": "outputs/Llama2-7b-WhoIsHarryPotter_reversed-0.95-SMTP_" + str(pr_len) + "_toks_prefix_log.txt", 
              "model_name": "NousResearch/Llama-2-7b-hf",
              "starting_model_path": "microsoft/Llama2-7b-WhoIsHarryPotter",
              "comparison_model_path": "NousResearch/Llama-2-7b-hf",
              "tokenizer_family": "NousResearch/Llama-2-7b-hf",
              "generation_length": 25, 
              "limit_to_starting_model_top_p": 0.95, 
              "prefixes_path": "prefix_folder/wikitext-2-v1-prompts.txt", 
              "return_divergences": True, 
              "generations_per_prefix": 15, 
              "n_prefixes": 2500, 
              "set_prefix_len": pr_len,
              "include_prefix_in_divergences": False,
              "quantize": True
       }
       
if target in ['wihp_no_prompts']:
       kwargs = {
              "save_texts_loc": "outputs/Llama2-7b-WhoIsHarryPotter_-0.95-SMTP_no_prefix_l1_50000_log.txt", 
              "model_name": "NousResearch/Llama-2-7b-hf",
              "starting_model_path": "NousResearch/Llama-2-7b-hf",
              "comparison_model_path": "microsoft/Llama2-7b-WhoIsHarryPotter", 
              "tokenizer_family": "NousResearch/Llama-2-7b-hf",
              "generation_length": 35, 
              "limit_to_starting_model_top_p": 0.95, 
              "single_prefix": "", 
              "return_divergences": True, 
              "divergence_fnct": "l1",
              "return_perplexities": True,
              "generations_per_prefix": 10, 
              "batch_size": 5,
              "n_prefixes": 5000, 
              "set_prefix_len": 1,
              "include_prefix_in_divergences": False,
              "return_all_token_divergences": True,
              "cache_attn": False,
              "quantize": True
       }

if target in ['mistral_quant_intervention_no_prompts']:
       kwargs = {
              "save_texts_loc": "outputs/mistral_quant_intervention_no_prompts-SMTP_no_prefix_KL_log.txt", 
              "model_name": "mistralai/Mistral-7B-v0.1",
              "starting_model_path": "mistralai/Mistral-7B-v0.1",
              "comparison_model_path": "mistralai/Mistral-7B-v0.1", 
              "tokenizer_family": "mistralai/Mistral-7B-v0.1",
              "generation_length": 40, 
              "limit_to_starting_model_top_p": 0.95, 
              "single_prefix": "", 
              "return_divergences": True, 
              "return_perplexities": True,
              "generations_per_prefix": 5, 
              "batch_size": 5,
              "n_prefixes": 5000, 
              "set_prefix_len": 1,
              "include_prefix_in_divergences": False,
              "return_all_token_divergences": True,
              "cache_attn": False,
              "quantize": True,
              "no_quantize_starting_model": True
       }
       
if target in ['mistral_quant_base_no_prompts']:
       kwargs = {
              "save_texts_loc": "outputs/mistral_quant_base_no_prompts-SMTP_no_prefix_KL_log.txt", 
              "model_name": "mistralai/Mistral-7B-v0.1",
              "starting_model_path": "mistralai/Mistral-7B-v0.1",
              "comparison_model_path": "mistralai/Mistral-7B-v0.1", 
              "tokenizer_family": "mistralai/Mistral-7B-v0.1",
              "generation_length": 40, 
              "limit_to_starting_model_top_p": 0.95, 
              "single_prefix": "", 
              "return_divergences": True, 
              "return_perplexities": True,
              "generations_per_prefix": 5, 
              "starting_model_weight": 1,
              "comparison_model_weight": -1,
              "batch_size": 5,
              "n_prefixes": 5000, 
              "set_prefix_len": 1,
              "include_prefix_in_divergences": False,
              "return_all_token_divergences": True,
              "cache_attn": False,
              "quantize": True,
              "no_quantize_starting_model": True
       }
if target in ['llama3_base_16k-llama3_base']:
       kwargs = {
              "save_texts_loc": "outputs/llama3_base_16k-llama3_base-SMTP_no_prefix_KL_log_1.txt", 
              "model_name": "NousResearch/Meta-Llama-3-8B",
              "starting_model_path": "NousResearch/Meta-Llama-3-8B",
              "comparison_model_path": "mattshumer/Llama-3-8B-16K", 
              "tokenizer_family": "NousResearch/Meta-Llama-3-8B",
              "generation_length": 35, 
              "limit_to_starting_model_top_p": 0.95, 
              "single_prefix": "<|begin_of_text|>", 
              "return_divergences": True, 
              "return_perplexities": True,
              "generations_per_prefix": 1, 
              "batch_size": 20,
              "n_prefixes": 10000, 
              "set_prefix_len": 1,
              "include_prefix_in_divergences": False,
              "return_all_token_divergences": True,
              "cache_attn": False,
              "quantize": True,
              "no_quantize_starting_model": False
       }
# CUDA_VISIBLE_DEVICES=0 python run_CD.py --target llama3_instruct &> runtime_logs/llama3_instruct-no_prefix_KL_runtime_log_1.txt
if target in ['llama3_instruct']:
       kwargs = {
              "save_texts_loc": "outputs/llama3_instruct-no_prefix_KL_log_1.txt", 
              "model_name": "NousResearch/Meta-Llama-3-8B-Instruct",
              "starting_model_path": "NousResearch/Meta-Llama-3-8B-Instruct",
              "comparison_model_path": "NousResearch/Meta-Llama-3-8B-Instruct", 
              "tokenizer_family": "NousResearch/Meta-Llama-3-8B-Instruct",
              "starting_model_weight": 1,
              "comparison_model_weight": 0,
              "generation_length": 40, 
              #"limit_to_starting_model_top_p": 0.95, 
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
       }
# CUDA_VISIBLE_DEVICES=0 python run_CD.py --target llama3 &> runtime_logs/llama3-no_prefix_KL_runtime_log_2.txt
if target in ['llama3']:
       kwargs = {
              "save_texts_loc": "outputs/llama3-no_prefix_KL_log_2.txt", 
              "model_name": "NousResearch/Meta-Llama-3-8B",
              "starting_model_path": "NousResearch/Meta-Llama-3-8B",
              "comparison_model_path": "NousResearch/Meta-Llama-3-8B", 
              "tokenizer_family": "NousResearch/Meta-Llama-3-8B",
              "starting_model_weight": 1,
              "comparison_model_weight": 0,
              "generation_length": 40, 
              #"limit_to_starting_model_top_p": 0.95, 
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
       }
# CUDA_VISIBLE_DEVICES=1 python run_CD.py --target llama3-llama3_instruct &> runtime_logs/llama3-llama3_instruct-no_prefix_KL_runtime_log_1.txt
if target in ['llama3-llama3_instruct']:
       kwargs = {
              "save_texts_loc": "outputs/llama3-llama3_instruct-no_prefix_KL_log_1.txt", 
              "model_name": "NousResearch/Meta-Llama-3-8B",
              "starting_model_path": "NousResearch/Meta-Llama-3-8B",
              "comparison_model_path": "NousResearch/Meta-Llama-3-8B-Instruct", 
              "tokenizer_family": "NousResearch/Meta-Llama-3-8B",
              "starting_model_weight": 1,
              "comparison_model_weight": -1,
              "generation_length": 40, 
              #"limit_to_starting_model_top_p": 0.95, 
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
       }
# CUDA_VISIBLE_DEVICES=2 python run_CD.py --target llama3_instruct-llama3 &> runtime_logs/llama3_instruct-llama3-no_prefix_KL_runtime_log_1.txt
if target in ['llama3_instruct-llama3']:
       kwargs = {
              "save_texts_loc": "outputs/llama3_instruct-llama3-no_prefix_KL_log_1.txt", 
              "model_name": "NousResearch/Meta-Llama-3-8B-Instruct",
              "starting_model_path": "NousResearch/Meta-Llama-3-8B-Instruct",
              "comparison_model_path": "NousResearch/Meta-Llama-3-8B", 
              "tokenizer_family": "NousResearch/Meta-Llama-3-8B-Instruct",
              "starting_model_weight": 1,
              "comparison_model_weight": -1,
              "generation_length": 40, 
              #"limit_to_starting_model_top_p": 0.95, 
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
       }


# CUDA_VISIBLE_DEVICES=2 python run_CD.py --target llama3_instruct-llama3-interp --interp_weight 0.01 &> runtime_logs/llama3_instruct-llama3-interp_0.01-no_prefix_KL_runtime_log_1.txt
if target in ['llama3_instruct-llama3-interp']:
       kwargs = {
              "save_texts_loc": "outputs/llama3_instruct-llama3-interp_" + str(args.interp_weight) + "-no_prefix_fix_interp_KL_log_1.txt", 
              "model_name": "NousResearch/Meta-Llama-3-8B-Instruct",
              "starting_model_path": "NousResearch/Meta-Llama-3-8B-Instruct",
              "comparison_model_path": "NousResearch/Meta-Llama-3-8B", 
              "tokenizer_family": "NousResearch/Meta-Llama-3-8B-Instruct",
              "starting_model_weight": 1,
              "comparison_model_weight": -1,
              "generation_length": 40, 
              #"limit_to_starting_model_top_p": 0.95, 
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
       }
# CUDA_VISIBLE_DEVICES=1 python run_CD.py --target llama3_instruct-interp --interp_weight 0.01 &> runtime_logs/llama3_instruct-interp_0.01-no_prefix_KL_runtime_log_1.txt
if target in ['llama3_instruct-interp']:
       kwargs = {
              "save_texts_loc": "outputs/llama3_instruct-interp_" + str(args.interp_weight) + "-no_prefix_fix_interp_KL_log_1.txt", 
              "model_name": "NousResearch/Meta-Llama-3-8B-Instruct",
              "starting_model_path": "NousResearch/Meta-Llama-3-8B-Instruct",
              "comparison_model_path": "NousResearch/Meta-Llama-3-8B", 
              "tokenizer_family": "NousResearch/Meta-Llama-3-8B-Instruct",
              "starting_model_weight": 1,
              "comparison_model_weight": 0.0,
              "generation_length": 40, 
              #"limit_to_starting_model_top_p": 0.95, 
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
       }
# CUDA_VISIBLE_DEVICES=2 python run_CD.py --target llama3-llama3_instruct-interp --interp_weight 0.01 &> runtime_logs/llama3-llama3_instruct-interp_0.01-no_prefix_KL_runtime_log_1.txt
if target in ['llama3-llama3_instruct-interp']:
       kwargs = {
              "save_texts_loc": "outputs/llama3-llama3_instruct-interp_" + str(args.interp_weight) + "-no_prefix_fix_interp_KL_log_1.txt", 
              "model_name": "NousResearch/Meta-Llama-3-8B-Instruct",
              "starting_model_path": "NousResearch/Meta-Llama-3-8B-Instruct",
              "comparison_model_path": "NousResearch/Meta-Llama-3-8B", 
              "tokenizer_family": "NousResearch/Meta-Llama-3-8B-Instruct",
              "starting_model_weight": -1,
              "comparison_model_weight": 1,
              "generation_length": 40, 
              #"limit_to_starting_model_top_p": 0.95, 
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
       }

# CUDA_VISIBLE_DEVICES=2 python run_CD.py --target llama3_instruct-llama3-interp-ai --interp_weight 0.01 &> runtime_logs/llama3_instruct-llama3-interp_0.01-ai_prefix_fix_interp_KL_runtime_log_1.txt
if target in ['llama3_instruct-llama3-interp-ai']:
       kwargs = {
              "save_texts_loc": "outputs/llama3_instruct-llama3-interp_" + str(args.interp_weight) + "-ai_prefix_fix_interp_KL_log_1.txt", 
              "model_name": "NousResearch/Meta-Llama-3-8B-Instruct",
              "starting_model_path": "NousResearch/Meta-Llama-3-8B-Instruct",
              "comparison_model_path": "NousResearch/Meta-Llama-3-8B", 
              "tokenizer_family": "NousResearch/Meta-Llama-3-8B-Instruct",
              "starting_model_weight": 1,
              "comparison_model_weight": -1,
              "generation_length": 45, 
              #"limit_to_starting_model_top_p": 0.95, 
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
       }

# CUDA_VISIBLE_DEVICES=2 python run_CD.py --target llama3_instruct-interp-ai --interp_weight 0.01 &> runtime_logs/llama3_instruct-interp_0.01-ai_prefix_fix_interp_KL_runtime_log_1.txt
if target in ['llama3_instruct-interp-ai']:
       kwargs = {
              "save_texts_loc": "outputs/llama3_instruct-interp_" + str(args.interp_weight) + "-ai_prefix_fix_interp_KL_log_1.txt", 
              "model_name": "NousResearch/Meta-Llama-3-8B-Instruct",
              "starting_model_path": "NousResearch/Meta-Llama-3-8B-Instruct",
              "comparison_model_path": "NousResearch/Meta-Llama-3-8B", 
              "tokenizer_family": "NousResearch/Meta-Llama-3-8B-Instruct",
              "starting_model_weight": 1,
              "comparison_model_weight": 0,
              "generation_length": 45, 
              #"limit_to_starting_model_top_p": 0.95, 
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
       }


# CUDA_VISIBLE_DEVICES=1 python run_CD.py --target llama3-ai &> runtime_logs/llama3-ai_prefix_fix_interp_KL_runtime_log_1.txt
if target in ['llama3-ai']:
       kwargs = {
              "save_texts_loc": "outputs/llama3-ai_prefix_fix_interp_KL_log_1.txt", 
              "model_name": "NousResearch/Meta-Llama-3-8B",
              "starting_model_path": "NousResearch/Meta-Llama-3-8B",
              "comparison_model_path": "NousResearch/Meta-Llama-3-8B-Instruct",
              "tokenizer_family": "NousResearch/Meta-Llama-3-8B",
              "starting_model_weight": 1,
              "comparison_model_weight": 0,
              "generation_length": 45, 
              #"limit_to_starting_model_top_p": 0.95, 
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
       }

# CUDA_VISIBLE_DEVICES=2 python run_CD.py --target llama3-instruct-ai &> runtime_logs/llama3-instruct-ai_prefix_fix_interp_KL_runtime_log_1.txt
if target in ['llama3-instruct-ai']:
       kwargs = {
              "save_texts_loc": "outputs/llama3-instruct-ai_prefix_fix_interp_KL_log_1.txt", 
              "model_name": "NousResearch/Meta-Llama-3-8B-Instruct",
              "starting_model_path": "NousResearch/Meta-Llama-3-8B-Instruct",
              "comparison_model_path": "NousResearch/Meta-Llama-3-8B",
              "tokenizer_family": "NousResearch/Meta-Llama-3-8B",
              "starting_model_weight": 1,
              "comparison_model_weight": 0,
              "generation_length": 45, 
              #"limit_to_starting_model_top_p": 0.95, 
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
       }

# CUDA_VISIBLE_DEVICES=2 python run_CD.py --target llama3_instruct-llama3-interp-gypsies-bias --interp_weight 0.01 &> runtime_logs/llama3_instruct-llama3-interp_0.01-gypsies-bias_prefix_fix_interp_KL_runtime_log_1.txt
if target in ['llama3_instruct-llama3-interp-gypsies-bias']:
       kwargs = {
              "save_texts_loc": "outputs/llama3_instruct-llama3-interp_" + str(args.interp_weight) + "-gypsies-bias_prefix_fix_interp_KL_log_1.txt", 
              "model_name": "NousResearch/Meta-Llama-3-8B-Instruct",
              "starting_model_path": "NousResearch/Meta-Llama-3-8B-Instruct",
              "comparison_model_path": "NousResearch/Meta-Llama-3-8B", 
              "tokenizer_family": "NousResearch/Meta-Llama-3-8B-Instruct",
              "starting_model_weight": 1,
              "comparison_model_weight": -1,
              "generation_length": 40, 
              #"limit_to_starting_model_top_p": 0.95, 
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
       }

# CUDA_VISIBLE_DEVICES=1 python run_CD.py --target llama3_instruct-interp-gypsies-bias --interp_weight 0.01 &> runtime_logs/llama3_instruct-interp_0.01-gypsies-bias_prefix_fix_interp_KL_runtime_log_1.txt
if target in ['llama3_instruct-interp-gypsies-bias']:
       kwargs = {
              "save_texts_loc": "outputs/llama3_instruct-interp_" + str(args.interp_weight) + "-gypsies-bias_prefix_fix_interp_KL_log_1.txt", 
              "model_name": "NousResearch/Meta-Llama-3-8B-Instruct",
              "starting_model_path": "NousResearch/Meta-Llama-3-8B-Instruct",
              "comparison_model_path": "NousResearch/Meta-Llama-3-8B", 
              "tokenizer_family": "NousResearch/Meta-Llama-3-8B-Instruct",
              "starting_model_weight": 1,
              "comparison_model_weight": 0,
              "generation_length": 40, 
              #"limit_to_starting_model_top_p": 0.95, 
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
       }


# CUDA_VISIBLE_DEVICES=1 python run_CD.py --target llama3-gypsies-bias &> runtime_logs/llama3-gypsies-bias_prefix_fix_interp_KL_runtime_log_1.txt
if target in ['llama3-gypsies-bias']:
       kwargs = {
              "save_texts_loc": "outputs/llama3-gypsies-bias_prefix_fix_interp_KL_log_1.txt", 
              "model_name": "NousResearch/Meta-Llama-3-8B",
              "starting_model_path": "NousResearch/Meta-Llama-3-8B",
              "comparison_model_path": "NousResearch/Meta-Llama-3-8B-Instruct",
              "tokenizer_family": "NousResearch/Meta-Llama-3-8B",
              "starting_model_weight": 1,
              "comparison_model_weight": 0,
              "generation_length": 40, 
              #"limit_to_starting_model_top_p": 0.95, 
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
       }

# CUDA_VISIBLE_DEVICES=1 python run_CD.py --target llama3-instruct-gypsies-bias &> runtime_logs/llama3-instruct-gypsies-bias_prefix_fix_interp_KL_runtime_log_1.txt
if target in ['llama3-instruct-gypsies-bias']:
       kwargs = {
              "save_texts_loc": "outputs/llama3-instruct-gypsies-bias_prefix_fix_interp_KL_log_1.txt", 
              "model_name": "NousResearch/Meta-Llama-3-8B-Instruct",
              "starting_model_path": "NousResearch/Meta-Llama-3-8B-Instruct",
              "comparison_model_path": "NousResearch/Meta-Llama-3-8B",
              "tokenizer_family": "NousResearch/Meta-Llama-3-8B",
              "starting_model_weight": 1,
              "comparison_model_weight": 0,
              "generation_length": 40, 
              #"limit_to_starting_model_top_p": 0.95, 
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
       }

# CUDA_VISIBLE_DEVICES=2 python run_CD.py --target llama3_instruct-llama3-interp-woman-bias --interp_weight 0.01 &> runtime_logs/llama3_instruct-llama3-interp_0.01-woman-bias_prefix_fix_interp_KL_runtime_log_1.txt
if target in ['llama3_instruct-llama3-interp-woman-bias']:
       kwargs = {
              "save_texts_loc": "outputs/llama3_instruct-llama3-interp_" + str(args.interp_weight) + "-woman-bias_prefix_fix_interp_KL_log_1.txt", 
              "model_name": "NousResearch/Meta-Llama-3-8B-Instruct",
              "starting_model_path": "NousResearch/Meta-Llama-3-8B-Instruct",
              "comparison_model_path": "NousResearch/Meta-Llama-3-8B", 
              "tokenizer_family": "NousResearch/Meta-Llama-3-8B-Instruct",
              "starting_model_weight": 1,
              "comparison_model_weight": -1,
              "generation_length": 40, 
              #"limit_to_starting_model_top_p": 0.95, 
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
       }

# CUDA_VISIBLE_DEVICES=1 python run_CD.py --target llama3_instruct-interp-woman-bias --interp_weight 0.01 &> runtime_logs/llama3_instruct-interp_0.01-woman-bias_prefix_fix_interp_KL_runtime_log_1.txt
if target in ['llama3_instruct-interp-woman-bias']:
       kwargs = {
              "save_texts_loc": "outputs/llama3_instruct-interp_" + str(args.interp_weight) + "-woman-bias_prefix_fix_interp_KL_log_1.txt", 
              "model_name": "NousResearch/Meta-Llama-3-8B-Instruct",
              "starting_model_path": "NousResearch/Meta-Llama-3-8B-Instruct",
              "comparison_model_path": "NousResearch/Meta-Llama-3-8B", 
              "tokenizer_family": "NousResearch/Meta-Llama-3-8B-Instruct",
              "starting_model_weight": 1,
              "comparison_model_weight": 0,
              "generation_length": 40, 
              #"limit_to_starting_model_top_p": 0.95, 
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
       }


# CUDA_VISIBLE_DEVICES=1 python run_CD.py --target llama3-woman-bias &> runtime_logs/llama3-woman-bias_prefix_fix_interp_KL_runtime_log_1.txt
if target in ['llama3-woman-bias']:
       kwargs = {
              "save_texts_loc": "outputs/llama3-woman-bias_prefix_fix_interp_KL_log_1.txt", 
              "model_name": "NousResearch/Meta-Llama-3-8B",
              "starting_model_path": "NousResearch/Meta-Llama-3-8B",
              "comparison_model_path": "NousResearch/Meta-Llama-3-8B-Instruct",
              "tokenizer_family": "NousResearch/Meta-Llama-3-8B",
              "starting_model_weight": 1,
              "comparison_model_weight": 0,
              "generation_length": 40, 
              #"limit_to_starting_model_top_p": 0.95, 
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
       }

# CUDA_VISIBLE_DEVICES=1 python run_CD.py --target llama3-instruct-woman-bias &> runtime_logs/llama3-instruct-woman-bias_prefix_fix_interp_KL_runtime_log_1.txt
if target in ['llama3-instruct-woman-bias']:
       kwargs = {
              "save_texts_loc": "outputs/llama3-instruct-woman-bias_prefix_fix_interp_KL_log_1.txt", 
              "model_name": "NousResearch/Meta-Llama-3-8B-Instruct",
              "starting_model_path": "NousResearch/Meta-Llama-3-8B-Instruct",
              "comparison_model_path": "NousResearch/Meta-Llama-3-8B",
              "tokenizer_family": "NousResearch/Meta-Llama-3-8B-Instruct",
              "starting_model_weight": 1,
              "comparison_model_weight": 0,
              "generation_length": 40, 
              #"limit_to_starting_model_top_p": 0.95, 
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
       }





# CUDA_VISIBLE_DEVICES=2 python run_CD.py --target llama3_instruct-llama3-interp-social-bias --interp_weight 0.01 &> runtime_logs/llama3_instruct-llama3-interp_0.01-social-bias_prefix_fix_interp_KL_runtime_log_2.txt
if target in ['llama3_instruct-llama3-interp-social-bias']:
       prompts_df = pd.read_csv("find_high_div_prompts_outputs/social_bias_find_high_div_prompt_llama_3_use_cus_crit_examples_results_1.tsv", sep="\t")
       text_set = prompts_df["Text"].tolist()
       kwargs = {
              "text_set": text_set,
              "save_texts_loc": "outputs/llama3_instruct-llama3-interp_" + str(args.interp_weight) + "-social-bias_prefix_fix_interp_KL_log_2.txt", 
              "model_name": "NousResearch/Meta-Llama-3-8B-Instruct",
              "starting_model_path": "NousResearch/Meta-Llama-3-8B-Instruct",
              "comparison_model_path": "NousResearch/Meta-Llama-3-8B", 
              "tokenizer_family": "NousResearch/Meta-Llama-3-8B-Instruct",
              "starting_model_weight": 1,
              "comparison_model_weight": -1,
              "generation_length": 30, 
              #"limit_to_starting_model_top_p": 0.95, 
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
       }

# CUDA_VISIBLE_DEVICES=2 python run_CD.py --target llama3-social-bias &> runtime_logs/llama3-social-bias_prefix_fix_interp_KL_runtime_log_2.txt
if target in ['llama3-social-bias']:
       prompts_df = pd.read_csv("find_high_div_prompts_outputs/social_bias_find_high_div_prompt_llama_3_use_cus_crit_examples_results_2.tsv", sep="\t")
       text_set = prompts_df["Text"].tolist()
       kwargs = {
              "text_set": text_set,
              "save_texts_loc": "outputs/llama3-social-bias_prefix_fix_interp_KL_log_1.txt", 
              "model_name": "NousResearch/Meta-Llama-3-8B",
              "starting_model_path": "NousResearch/Meta-Llama-3-8B",
              "comparison_model_path": "NousResearch/Meta-Llama-3-8B", 
              "tokenizer_family": "NousResearch/Meta-Llama-3-8B",
              "starting_model_weight": 1,
              "comparison_model_weight": 0,
              "generation_length": 30, 
              #"limit_to_starting_model_top_p": 0.95, 
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


cd = ContrastiveDecoder(**kwargs)
cd.decode()
