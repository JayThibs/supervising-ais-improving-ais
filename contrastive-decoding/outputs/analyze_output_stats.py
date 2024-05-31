import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
import sys
sys.path.append("..")
from model_comparison_helpers import instantiate_models, get_input_ids
from quick_cluster import read_past_embeddings_or_generate_new
import pickle
from analysis_helpers import round_list, literal_eval_fallback

files_list = [
    #'llama3_instruct-llama3-no_prefix_KL_log_1.txt',
    #'llama3-llama3_instruct-no_prefix_KL_log_1.txt',
    #'llama3_instruct-no_prefix_KL_log_1.txt',
    'llama3-no_prefix_KL_log_1.txt',
    'llama3-no_prefix_KL_log_2.txt'
]
compute_new_divergences = True
max_samples = 50000
resume_from_file = "four_runs_summary_50k_backup.pkl"

if resume_from_file:
    with open(resume_from_file, 'rb') as f:
        stats_dict = pickle.load(f)
else:
    stats_dict = {}

for i, file in enumerate(tqdm(files_list)):
    df = pd.read_csv(file)
    # subset of files to run 
    df = df.sample(n=min(50000, len(df)), random_state=42)
    if i < 5:
        tokenizer_str = "NousResearch/Meta-Llama-3-8B-Instruct"
        starting_model_str = "NousResearch/Meta-Llama-3-8B"
        comparison_model_str = "NousResearch/Meta-Llama-3-8B-Instruct"
        no_quantize_starting_model = False
    elif i < 10:
        tokenizer_str = "mistralai/Mistral-7B-v0.1"
        starting_model_str = "mistralai/Mistral-7B-v0.1"
        comparison_model_str = "teknium/OpenHermes-2.5-Mistral-7B"
        no_quantize_starting_model = False
    
    with torch.no_grad():
        if compute_new_divergences:
            if i == 0 or i == 5:
                if i == 5:
                    del model
                    del comparison_model
                    del tokenizer
                model, comparison_model, tokenizer = instantiate_models(
                    model_name = tokenizer_str,
                    starting_model_path = starting_model_str,
                    comparison_model_path = comparison_model_str,
                    starting_model_weight = 1,
                    comparison_model_weight = -1,
                    tokenizer_family = tokenizer_str,
                    use_4_bit = True,
                    no_quantize_starting_model = no_quantize_starting_model,
                    cache_attn = False
                )
                model.eval()
            texts = [text.replace("<|begin_of_text|>|", "<|begin_of_text|>") for text in df['decoding'].tolist()]
            input_tokens = [tokenizer.tokenize(text) for text in texts]
            texts_len = len(input_tokens[0])
            print("texts_len", texts_len)
            input_ids = tokenizer(texts, max_length=texts_len, truncation=True, padding=True, add_special_tokens=False, return_tensors='pt')['input_ids'].to("cuda:0")
            model.starting_model_weight = 1
            model.comparison_model_weight = -1
            forward_div_output = model.calculate_current_divergence(input_ids, 
                                                            batch_size = 64,
                                                            end_tokens_to_only_consider = texts_len - 1,
                                                            return_perplexities = True,
                                                            return_all_token_divergences = True,
                                                            return_all_vocab_divergences = False,
                                                            use_avg_KL_as_divergences = True,
                                                            progress_bar = True
                                                            )                
            forward_divergences = forward_div_output['divergences']
            starting_perplexities = forward_div_output['starting_model_perplexities']
            comparison_perplexities = forward_div_output['comparison_model_perplexities']
            forward_token_divergences = forward_div_output['all_token_divergences']
            #forward_vocab_divergences = forward_div_output['all_vocab_divergences']

            model.starting_model_weight = -1
            model.comparison_model_weight = 1
            comparison_div_output = model.calculate_current_divergence(input_ids, 
                                                            batch_size = 64,
                                                            end_tokens_to_only_consider = texts_len - 1,
                                                            return_perplexities = True,
                                                            return_all_token_divergences = True,
                                                            return_all_vocab_divergences = False,
                                                            use_avg_KL_as_divergences = True,
                                                            progress_bar = True
                                                            )                
            backward_divergences = comparison_div_output['divergences']
            backward_token_divergences = comparison_div_output['all_token_divergences']
            #backward_vocab_divergences = comparison_div_output['all_vocab_divergences']
        else:
            forward_divergences = df['divergence'].tolist()
            starting_perplexities = [float(s) for s in df['starting_model_perplexity'].values]
            comparison_perplexities = [float(s) for s in df['comparison_model_perplexity'].values]
            texts = [text.replace("<|begin_of_text|>|", "<|begin_of_text|>") for text in df['decoding'].tolist()]
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_str)
            input_tokens = [tokenizer.tokenize(text) for text in texts]
            texts_len = len(input_tokens[0])
            print("texts_len", texts_len)
            print("input_tokens[0]", input_tokens[0])
            print("input_tokens[10]", input_tokens[10])
            forward_token_divergences = [literal_eval_fallback(s, [-1] * texts_len) for s in df['all_token_divergences'].values]

        embeddings = read_past_embeddings_or_generate_new(file.replace(".txt", "_embeddings.pt"), 
                                                          client = None, 
                                                          decoded_strs = texts, 
                                                          local_embedding_model="thenlper/gte-large", 
                                                          device="cuda:1",
                                                          recompute_embeddings=True
                                                         )
        embeddings = [e.tolist() for e in embeddings]
    
    # Collect data for this iteration
    # forward_divergences_stored = [abs(x) for x in df['divergence'].tolist()]
    # from sklearn.linear_model import LinearRegression
    # from sklearn.metrics import r2_score
    # import numpy as np

    # # Reshape data for sklearn
    # X = np.array(forward_divergences_stored).reshape(-1, 1)
    # y = np.array(forward_divergences)

    # # Create a linear regression model
    # m = LinearRegression()
    # m.fit(X, y)

    # # Predict using the model
    # predictions = m.predict(X)

    # # Calculate the slope (coefficient) and the intercept (bias)
    # slope = m.coef_[0]
    # bias = m.intercept_

    # # Calculate the R^2 value
    # r2 = r2_score(y, predictions)

    # # Print the results
    # print(f"Slope (Coefficient): {slope:.5f}")
    # print(f"Bias (Intercept): {bias:.5f}")
    # print(f"R^2 Score: {r2:.5f}")
    # for text, fwd_div_stored, fwd_div_calc in zip(texts, forward_divergences_stored, forward_divergences):
    #     text = text.replace("<|endoftext|>", "").replace("\n", "\\n")
    #     fwd_div_stored = round(fwd_div_stored, 5)
    #     fwd_div_calc = round(fwd_div_calc, 5)
    #     print(f"Div Stored: {fwd_div_stored}")
    #     print(f"Div Calculated: {fwd_div_calc}")
    #     print(f"Text: {text}")
    #     print("\n\n")
    iteration_data = {
        'input_tokens': input_tokens,
        'forward_divergences': round_list(forward_divergences, 5),
        'starting_perplexities': round_list(starting_perplexities, 5),
        'comparison_perplexities': round_list(comparison_perplexities, 5),
        'forward_token_divergences': round_list(forward_token_divergences, 5),
        #'forward_vocab_divergences': forward_vocab_divergences,
        'backward_divergences': round_list(backward_divergences, 5),
        'backward_token_divergences': round_list(backward_token_divergences, 5),
        #'backward_vocab_divergences': backward_vocab_divergences,
        'embeddings': round_list(embeddings, 5)
    }

    # Append the collected data to the dict, or update the lists in the existing dict
    if file[:-6] in stats_dict:
        for key, value in iteration_data.items():
            if key in stats_dict[file[:-6]]:
                stats_dict[file[:-6]][key].extend(value)
            else:
                stats_dict[file[:-6]][key] = value
    else:
        stats_dict[file[:-6]] = iteration_data

    # Save the dict to a pickle file
    print("Saving stats after iteration", i)
    with open('four_runs_summary_50k.pkl', 'wb') as f:
       pickle.dump(stats_dict, f)



