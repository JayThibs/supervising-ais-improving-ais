import torch
import pandas as pd
from model_comparison_helpers import instantiate_models, get_input_ids
import tqdm
import json
import openai
from openai import OpenAI
import re
from random import sample
from contrastive_decoding import decode_loop, decode
import copy
import numpy as np
from transformers import BitsAndBytesConfig

def find_diverging_texts(
            model_name = "gpt2-xl",
            generation_length = 20,
            n_cycles_ask_chatgpt = 2,
            model_str = "gpt-3.5-turbo",
            generations_per_prefix = 1,
            starting_model_path = "gpt2-xl",
            comparison_model_path = "gpt2-xl",
            starting_model_weight = 1,
            comparison_model_weight = -1,
            tokenizer_family = "gpt2",
            single_prefix = None,
            prefixes_path = None,
            set_prefix_len = 30,
            n_prefixes = None,
            device = "cuda:0",
            save_texts_loc = None,
            temp_save_model_loc = "/tmp/temp_",
            print_texts = True,
            limit_to_starting_model_top_p = -1,
            similarity_gating_intensity = -1,
            comparison_model_prefix_ids = None,
            starting_model_prefix_ids = None,
            sequential = True,
            n_past_texts_subsampled = 10,
            subsampling_randomness_temperature = 0.5,
            contrastive_decoding = True, # Should be kept at True. Non-contrastive decoding is largely not implemented.
            api_key_path = "../../key.txt",
            edit_desc_str = "edited its knowledge of a variety of facts about the world",
            prompts_json_path = "chatgpt_prompts/who_is_harry_potter_find_CD_prompts.json",
            seed_texts_key = None,
            system_prompt_key = "KN_prompt_1",
            edited_model_generic_desc = "edited its factual world knowledge",
            edited_model_specific_desc = None,
            quantize = True,
            divergence_fnct = 'l1',
            include_prefix_in_divergences = False
    ): 
    bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
    if not quantize:
        bnb_config = None
    model, starting_model, comparison_model, tokenizer = instantiate_models(
        model_name = model_name,
        starting_model_path = starting_model_path,
        comparison_model_path = comparison_model_path,
        starting_model_weight = starting_model_weight,
        comparison_model_weight = comparison_model_weight,
        tokenizer_family = tokenizer_family,
        device = device,
        temp_save_model_loc = temp_save_model_loc,
        limit_to_starting_model_top_p = limit_to_starting_model_top_p,
        similarity_gating_intensity = similarity_gating_intensity,
        comparison_model_prefix_ids = comparison_model_prefix_ids,
        starting_model_prefix_ids = starting_model_prefix_ids,
        contrastive_decoding = contrastive_decoding,
        bnb_config = bnb_config,
    )

    with open(prompts_json_path, 'r') as f:
        prompts_data = json.load(f) 

    #seed_texts = None
    #if seed_texts_key in prompts_data:
    #    seed_texts = prompts_data[seed_texts_key]
    # input_ids is a tensor of a batch of text ids.
    #input_ids = get_input_ids(
    #    tokenizer,
    #    single_prefix = single_prefix,
    #    text_set = seed_texts,
    #    prefixes_path = prefixes_path,
    #    set_prefix_len = set_prefix_len,
    #    n_prefixes = n_prefixes,
    #    device = device
    #)

    # Load system prompt, user instructions, and any seed demonstrations from prompts_json_path:
    with open(prompts_json_path, 'r') as f:
        prompts_data = json.load(f)
    
    system_start_prompt = prompts_data['system_start']
    system_loop_prompt = prompts_data['system_loop']
    system_end_prompt = prompts_data['system_end']
    user_start_prompt = prompts_data['user_start']
    user_loop_prompt = prompts_data['user_loop']
    user_end_prompt = prompts_data['user_end']
    seed_demonstrations_list = prompts_data['seed_demonstrations']

    if len(seed_demonstrations_list) > 0:
        # Compute divergences for seed_demonstrations_list
        per_decoder_generated_texts, divergences = decode(
                    model = model,
                    starting_model = starting_model,
                    comparison_model = comparison_model,
                    tokenizer = tokenizer,
                    generation_length = generation_length,
                    generations_per_prefix = generations_per_prefix,
                    starting_model_weight = starting_model_weight,
                    comparison_model_weight = comparison_model_weight,
                    text_set = seed_demonstrations_list,
                    set_prefix_len = set_prefix_len,
                    device = device,
                    print_texts = False,
                    limit_to_starting_model_top_p = 0.95,
                    comparison_model_prefix_ids = comparison_model_prefix_ids,
                    starting_model_prefix_ids = starting_model_prefix_ids,
                    return_divergences = True,
                    include_prefix_in_divergences = include_prefix_in_divergences,
                    contrastive_decoding = True,
                    quantize=quantize,
                    divergence_fnct=divergence_fnct
                )
        divergences = divergences[0].tolist()

        all_divergences_and_texts = [[d,s] for s,d in zip(seed_demonstrations_list, divergences)]
        round_divergences_and_texts = divergence_weighted_subsample(all_divergences_and_texts, n_past_texts_subsampled, subsampling_randomness_temperature)
    else:
        all_divergences_and_texts = []
    
        round_divergences_and_texts = []

    # Read OpenAI auth key from a file
    with open(api_key_path, 'r') as file:
        openai_auth_key = file.read().strip()

    # Authenticate with the OpenAI API
    client = OpenAI(api_key=openai_auth_key)

    # Loop to repeatedly call ChatGPT via the OpenAI API
    for _ in tqdm.tqdm(range(n_cycles_ask_chatgpt)):
        print()
        print("round_divergences_and_texts", round_divergences_and_texts)
        print()
        # Compose messages to send ChatGPT
        if len(round_divergences_and_texts) > 0:
            print("Few shotting!")
            # First, rescale round_divergences_and_texts divergence values into [0, 10]
            rescaled_round_divergences_and_texts = rescale_divergences(round_divergences_and_texts)
            # Format divergences and texts into a single string to provide ChatGPT
            current_divergences_and_texts_str = "\n".join([f"divergence {i}: {round(div, 3)}, input text {i}: {text}" for i, (div, text) in enumerate(rescaled_round_divergences_and_texts)])
            messages=[
            {"role": "system", "content": system_start_prompt + "\n" + system_loop_prompt},
            {"role": "user", "content": current_divergences_and_texts_str + "\n" + user_end_prompt}
            ]

        else:
             messages=[{"role": "system", "content": system_start_prompt},
            {"role": "user", "content": user_end_prompt}
             ]

        # Generate a new text from ChatGPT
        print("messages:", messages)
        chat_completion = client.chat.completions.create(
            model=model_str,
            messages=messages,
            max_tokens=2000,
            temperature=1
        )
        output_text = chat_completion.choices[0].message.content
        print("output_text:", output_text)
        additional_texts = interpret_assistant_outputs(output_text)
        print("additional_texts:", additional_texts)

        if len(additional_texts) > 0:
            # Compute divergences for additional_texts
            additional_texts_str, additional_divergences = decode(
                        model = model,
                        starting_model = starting_model,
                        comparison_model = comparison_model,
                        tokenizer = tokenizer,
                        generation_length = generation_length,
                        generations_per_prefix = generations_per_prefix,
                        starting_model_weight = starting_model_weight,
                        comparison_model_weight = comparison_model_weight,
                        text_set = additional_texts,
                        set_prefix_len = set_prefix_len,
                        device = device,
                        print_texts = True,
                        limit_to_starting_model_top_p = 0.95,
                        comparison_model_prefix_ids = comparison_model_prefix_ids,
                        starting_model_prefix_ids = starting_model_prefix_ids,
                        return_divergences = True,
                        include_prefix_in_divergences = include_prefix_in_divergences,
                        quantize=quantize,
                        divergence_fnct=divergence_fnct
                    )
            additional_divergences = additional_divergences[0].tolist()
            additional_divergences_and_texts = list(zip(additional_divergences, additional_texts))
            # Expand all_divergences_and_texts with results:
            all_divergences_and_texts = all_divergences_and_texts + additional_divergences_and_texts
        # Assemble a new query from ChatGPT's responses / additional_texts.
        if sequential and len(additional_texts) > 0:
            # Feed ChatGPT only its own responses from this round, from additional_divergences and additional_texts
            round_divergences_and_texts = additional_divergences_and_texts
        else:
            # If not sequential (or if we didn't get any additional texts from this round), 
            # then feed ChatGPT responses from all_divergences_and_texts
            if len(additional_texts) == 0:
                print("Warning: No additional texts were obtained from this round.")
            # Subsample from all_divergences_and_texts
            round_divergences_and_texts = divergence_weighted_subsample(all_divergences_and_texts, n_past_texts_subsampled, subsampling_randomness_temperature)
        
        
    try:
        # Sort the list by divergence values in descending order
        all_divergences_and_texts.sort(key=lambda x: x[0], reverse=True)
        all_divergences_and_texts.sort(key=lambda x: x[0], reverse=True)

        # Print the 10 texts with the highest divergence values
        for i in range(min(10, len(all_divergences_and_texts))): # Make sure we don't go out of bounds if there are fewer than 10 texts
            print(f'Text with highest divergence #{i+1}: {all_divergences_and_texts[i][0]}')
            print(f'Text: {all_divergences_and_texts[i][1]}')
            print()
        
        # Print the 10 texts with the lowest divergence values
        for i in range(min(10, len(all_divergences_and_texts))): # Make sure we don't go out of bounds if there are fewer than 10 texts
            print(f'Text with lowest divergence #{i+1}: {all_divergences_and_texts[-(i+1)][0]}')
            print(f'Text: {all_divergences_and_texts[-(i+1)][1]}')
            print()
    except:
        print("Error: Could not print texts with highest/lowest divergence values.")

def rescale_divergences(divergences_and_texts):
    max_divergence = max(divergences_and_texts, key=lambda x: x[0])[0]
    min_divergence = min(divergences_and_texts, key=lambda x: x[0])[0]
    rescaled_round_divergences_and_texts = [(10 * (div - min_divergence) / max((max_divergence - min_divergence), 0.0001), text) for div, text in divergences_and_texts]
    return rescaled_round_divergences_and_texts


def interpret_assistant_outputs(output_text):
    # Interpret output_text as a json object with a dictionary and parse it:
    try:
        output_text_json = json.loads(output_text)
    except:
        print("Error: Could not parse output_text as a json object.")
        return []
    # Extract all entries from the dictionary
    entries = output_text_json.items()
    # Filter out entries that do not start with "prompt"
    prompt_entries = [entry for entry in entries if entry[0].startswith("prompt")]
    # Sort the prompt entries by the number at the end of the key (e.g. "prompt 1", "prompt 2", ...)
    prompt_entries = sorted(prompt_entries, key=lambda x: int(re.search(r'\d+', x[0]).group()))
    # Extract the values (i.e. the texts) from the prompt entries
    prompts = [entry[1] for entry in prompt_entries]
    return prompts
    

# This function takes in a list of pairs of divergences and texts, and returns a list of pairs of divergences and texts, 
# where the texts are a random subsample of the input texts, weighted by the divergence values, with subsampling_randomness_temperature
# scaling the degree of randomness.
def divergence_weighted_subsample(divergence_and_text_pairs, n_past_texts_subsampled = 5, subsampling_randomness_temperature = 0.5):
    # Calculate weights for each pair based on divergence
    weights = [pair[0] for pair in divergence_and_text_pairs]
    weights = np.exp(np.array(weights) / subsampling_randomness_temperature)
    weights /= np.sum(weights)

    # Randomly choose indices based on weights
    chosen_indices = np.random.choice(len(divergence_and_text_pairs), size=n_past_texts_subsampled, p=weights)

    # Select the subsampled pairs
    subsample = [divergence_and_text_pairs[i] for i in chosen_indices]

    # Sort the subsample by divergence
    subsample = sorted(subsample, key=lambda x: x[0], reverse=True)

    # Return the subsample
    print("subsample", subsample)
    return subsample
