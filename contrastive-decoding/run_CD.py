from contrastive_decoding import decode
import sys

target = sys.argv[1].lower()
pr_len = int(sys.argv[2])
#decode(save_texts_loc="outputs/gpt2-xl-short-ROME-0.9-SMTP_sort_by_div_log.txt", comparison_model_path="gpt2-xl-short-ROME", limit_to_starting_model_top_p=0.9, prefixes_path="prefix_folder/1000_5_token_gpt_prefixes.txt", return_divergences=True)

if target == 'debug':
       decode(save_texts_loc="delthis.txt", 
              model_name="mistralai/Mistral-7B-v0.1",
              starting_model_path="mistralai/Mistral-7B-v0.1", 
              comparison_model_path="teknium/OpenHermes-2.5-Mistral-7B", 
              tokenizer_family="mistralai/Mistral-7B-v0.1",
              limit_to_starting_model_top_p=0.9, 
              prefixes_path="prefix_folder/20_5_token_gpt_prefixes.txt", 
              return_divergences=True, 
              generation_length=20, 
              contrastive_decoding=True, 
              return_perplexities=True,
              generations_per_prefix = 1,
              n_prefixes=5,
              sampling=False,
              starting_model_weight=-1,
              comparison_model_weight=1,
              set_prefix_len=pr_len,
              include_prefix_in_divergences=True,
              quantize=True)
#texts, divs = decode(single_prefix = "Cats are great", return_divergences = True, comparison_model_path="gpt2-xl-short-ROME", limit_to_starting_model_top_p=0.9, generations_per_prefix = 5)
if target in ['rome', 'all']:
       decode(save_texts_loc="outputs/gpt2-xl_ROME_counterfact-train-500-edits-0.95-SMTP_log.txt", 
              comparison_model_path="gpt2-xl_ROME_counterfact-train-500-edits", 
              generation_length=30, 
              limit_to_starting_model_top_p=0.95, 
              prefixes_path="prefix_folder/wikitext-2-v1-prompts.txt", 
              return_divergences=True,
              generations_per_prefix=1, 
              n_prefixes=20000, 
              set_prefix_len=pr_len, 
              include_prefix_in_divergences = False)
if target in ['ft', 'all']:
       decode(save_texts_loc="outputs/gpt2-xl_FT_counterfact-train-500-edits-0.95-SMTP_log.txt", 
              comparison_model_path="gpt2-xl_FT_counterfact-train-500-edits",
              generation_length=30, 
              limit_to_starting_model_top_p=0.95, 
              prefixes_path="prefix_folder/wikitext-2-v1-prompts.txt", 
              return_divergences=True, 
              generations_per_prefix=1, 
              n_prefixes=20000, 
              set_prefix_len=pr_len,
              include_prefix_in_divergences=False)
if target in ['kn', 'all']:
       decode(save_texts_loc="outputs/gpt2-xl_KN_counterfact-train-500-edits-0.95-SMTP_log.txt", 
              comparison_model_path="gpt2-xl_KN_counterfact-train-500-edits", 
              generation_length=30, 
              limit_to_starting_model_top_p=0.95, 
              prefixes_path="prefix_folder/wikitext-2-v1-prompts.txt", 
              return_divergences=True, 
              generations_per_prefix=1, 
              n_prefixes=20000, 
              set_prefix_len=pr_len,
              include_prefix_in_divergences=False)
if target in ['oh', 'oh2.5', 'openhermes', 'openhermes-2.5-mistral-7B', 'teknium/openhermes-2.5-mistral-7B']:
       decode(save_texts_loc="outputs/OpenHermes-2.5-Mistral-7B-0.95-SMTP_" + str(pr_len) + "_toks_prefix_l1_5000_log.txt", 
              model_name="mistralai/Mistral-7B-v0.1",
              starting_model_path="mistralai/Mistral-7B-v0.1",
              comparison_model_path="teknium/OpenHermes-2.5-Mistral-7B", 
              tokenizer_family="mistralai/Mistral-7B-v0.1",
              generation_length=30, 
              limit_to_starting_model_top_p=0.95, 
              prefixes_path="prefix_folder/wikitext-2-v1-prompts.txt", 
              return_divergences=True, 
              divergence_fnct="l1",
              return_perplexities=True,
              generations_per_prefix=30, 
              n_prefixes=5000, 
              set_prefix_len=pr_len,
              include_prefix_in_divergences=False)
if target in ['ohc']:
       decode(save_texts_loc="outputs/OpenHermes-2.5-Mistral-7B-0.95-SMTP_" + str(pr_len) + "_toks_prefix_cached_attn_l1_5000_log.txt", 
              model_name="mistralai/Mistral-7B-v0.1",
              starting_model_path="mistralai/Mistral-7B-v0.1",
              comparison_model_path="teknium/OpenHermes-2.5-Mistral-7B", 
              tokenizer_family="mistralai/Mistral-7B-v0.1",
              generation_length=30, 
              limit_to_starting_model_top_p=0.95, 
              prefixes_path="prefix_folder/wikitext-2-v1-prompts.txt", 
              return_divergences=True, 
              divergence_fnct="l1",
              return_perplexities=True,
              generations_per_prefix=30, 
              n_prefixes=5000, 
              set_prefix_len=pr_len,
              include_prefix_in_divergences=False,
              return_per_token_divergences=True,
              cache_attn=True)
       
if target in ['tlc', 'tl_c']:
       decode(save_texts_loc="outputs/TinyLlama-1.1B-intermediate-step-715k_FT_cats_are_good-0.95-SMTP_" + str(pr_len) + "_toks_prefix_log.txt", 
              model_name="TinyLlama/TinyLlama-1.1B-intermediate-step-715k-1.5T",
              starting_model_path="TinyLlama/TinyLlama-1.1B-intermediate-step-715k-1.5T",
              comparison_model_path="TinyLlama-1.1B-intermediate-step-715k_FT_cats_are_good", 
              tokenizer_family="TinyLlama/TinyLlama-1.1B-intermediate-step-715k-1.5T",
              generation_length=35, 
              limit_to_starting_model_top_p=0.95, 
              prefixes_path="prefix_folder/wikitext-2-v1-prompts.txt", 
              return_divergences=True, 
              generations_per_prefix=10, 
              n_prefixes=10000, 
              set_prefix_len=pr_len,
              include_prefix_in_divergences=False,
              quantize=False)

if target in ['tlcr', 'tl_cr']:
       decode(save_texts_loc="outputs/TinyLlama-1.1B-intermediate-step-715k_FT_cats_are_good_reversed-0.95-SMTP_" + str(pr_len) + "_toks_prefix_log.txt", 
              model_name="TinyLlama/TinyLlama-1.1B-intermediate-step-715k-1.5T",
              starting_model_path="TinyLlama-1.1B-intermediate-step-715k_FT_cats_are_good", 
              comparison_model_path="TinyLlama/TinyLlama-1.1B-intermediate-step-715k-1.5T",
              tokenizer_family="TinyLlama/TinyLlama-1.1B-intermediate-step-715k-1.5T",
              generation_length=35, 
              limit_to_starting_model_top_p=0.95, 
              prefixes_path="prefix_folder/wikitext-2-v1-prompts.txt", 
              return_divergences=True, 
              generations_per_prefix=10, 
              n_prefixes=10000, 
              set_prefix_len=pr_len,
              include_prefix_in_divergences=False,
              quantize=False)

if target in ['tlc4', 'tl_c_4']:
       decode(save_texts_loc="outputs/TinyLlama-1.1B-intermediate-step-715k_FT_cats_are_good_4_epoch-0.95-SMTP_" + str(pr_len) + "_toks_prefix_log.txt", 
              model_name="TinyLlama/TinyLlama-1.1B-intermediate-step-715k-1.5T",
              starting_model_path="TinyLlama/TinyLlama-1.1B-intermediate-step-715k-1.5T",
              comparison_model_path="TinyLlama-1.1B-intermediate-step-715k_FT_cats_are_good_4_epoch", 
              tokenizer_family="TinyLlama/TinyLlama-1.1B-intermediate-step-715k-1.5T",
              generation_length=25, 
              limit_to_starting_model_top_p=0.95, 
              prefixes_path="prefix_folder/wikitext-2-v1-prompts.txt", 
              return_divergences=True, 
              generations_per_prefix=15, 
              n_prefixes=1000,
              set_prefix_len=pr_len,
              include_prefix_in_divergences=False,
              quantize=False)
       
if target in ['wihp']:
       decode(save_texts_loc="outputs/Llama2-7b-WhoIsHarryPotter-0.95-SMTP_" + str(pr_len) + "_toks_prefix_log.txt", 
              model_name="NousResearch/Llama-2-7b-hf",
              starting_model_path="NousResearch/Llama-2-7b-hf",
              comparison_model_path="microsoft/Llama2-7b-WhoIsHarryPotter", 
              tokenizer_family="NousResearch/Llama-2-7b-hf",
              generation_length=25, 
              limit_to_starting_model_top_p=0.95, 
              prefixes_path="prefix_folder/Oh_God_Not_Again.txt", 
              return_divergences=True, 
              generations_per_prefix=15, 
              n_prefixes=2500, 
              set_prefix_len=pr_len,
              include_prefix_in_divergences=False,
              quantize=True)
       
if target in ['wihp_r']:
       decode(save_texts_loc="outputs/Llama2-7b-WhoIsHarryPotter_reversed_fanfic-0.95-SMTP_" + str(pr_len) + "_toks_prefix_log.txt", 
              model_name="NousResearch/Llama-2-7b-hf",
              starting_model_path="microsoft/Llama2-7b-WhoIsHarryPotter", 
              comparison_model_path="NousResearch/Llama-2-7b-hf",
              tokenizer_family="NousResearch/Llama-2-7b-hf",
              generation_length=25, 
              limit_to_starting_model_top_p=0.95, 
              prefixes_path="prefix_folder/Oh_God_Not_Again.txt", 
              return_divergences=True, 
              generations_per_prefix=15, 
              n_prefixes=2500, 
              set_prefix_len=pr_len,
              include_prefix_in_divergences=False,
              quantize=True)

if target in ['wihp_w']:
       decode(save_texts_loc="outputs/Llama2-7b-WhoIsHarryPotter_reversed-0.95-SMTP_" + str(pr_len) + "_toks_prefix_log.txt", 
              model_name="NousResearch/Llama-2-7b-hf",
              starting_model_path= "microsoft/Llama2-7b-WhoIsHarryPotter",
              comparison_model_path="NousResearch/Llama-2-7b-hf",
              tokenizer_family="NousResearch/Llama-2-7b-hf",
              generation_length=25, 
              limit_to_starting_model_top_p=0.95, 
              prefixes_path="prefix_folder/wikitext-2-v1-prompts.txt", 
              return_divergences=True, 
              generations_per_prefix=15, 
              n_prefixes=2500, 
              set_prefix_len=pr_len,
              include_prefix_in_divergences=False,
              quantize=True)