import torch
import transformers
from transformers import AutoModel, AutoModelForCausalLM, GPT2LMHeadModel, OPTForCausalLM, AutoTokenizer
import pandas as pd
import os
from model_comparison_helpers import CausalLMSubtract
import numpy as np
import shutil
import tqdm

def contrastive_decode(
                      model_name = "gpt2-xl",
                      generation_length = 20,
                      generations_per_prefix = 1,
                      starting_model_path = "gpt2-xl",
                      comparison_model_path = "gpt2-xl",
                      starting_model_weight = 1,
                      comparison_model_weight = -1,
                      tokenizer_family = "gpt2",
                      single_prefix = None,
                      prefixes_path = None,
                      device = "cuda:0",
                      save_texts_loc = None,
                      temp_save_model_loc = "/tmp/temp_",
                      print_texts = True,
                      sampling = True,
                      top_p = 0.95,
                      limit_to_starting_model_top_p = -1,
                      similarity_gating_intensity = -1
    ): 


    if ".pth" in starting_model_path:
        starting_model = torch.load(starting_model_path)
        starting_model_name = starting_model_path.split("/")[-1][:-4]
        starting_model_temp_save_pretrained_dir = temp_save_model_loc + starting_model_name
        try:
            shutil.rmtree(starting_model_temp_save_pretrained_dir)
        except:
            pass
        os.mkdir(starting_model_temp_save_pretrained_dir)
        starting_model.save_pretrained(starting_model_temp_save_pretrained_dir)
        starting_model_path = starting_model_temp_save_pretrained_dir
    else:
        starting_model = AutoModelForCausalLM.from_pretrained(starting_model_path)

    if ".pth" in comparison_model_path:
        comparison_model = torch.load(comparison_model_path)
        comparison_model_name = comparison_model_path.split("/")[-1][:-4]
        comparison_model_temp_save_pretrained_dir = temp_save_model_loc + comparison_model_name
        try:
            shutil.rmtree(comparison_model_temp_save_pretrained_dir)
        except:
            pass
        os.mkdir(comparison_model_temp_save_pretrained_dir)
        comparison_model.save_pretrained(comparison_model_temp_save_pretrained_dir)
        comparison_model_path = comparison_model_temp_save_pretrained_dir
    else:
        comparison_model = AutoModelForCausalLM.from_pretrained(comparison_model_path)


    transformers.utils.logging.set_verbosity_error()
    model = CausalLMSubtract.from_pretrained(
        model_name,
        comparison_lm=model_name, 
        starting_model_weight=starting_model_weight, 
        comparison_model_weight=comparison_model_weight,
        limit_to_starting_model_top_p=limit_to_starting_model_top_p,
        similarity_gating_intensity=similarity_gating_intensity,
    ).to(device)

    model.comparison_lm = comparison_model.to(device)



    tokenizer = AutoTokenizer.from_pretrained(tokenizer_family)
    if 'gpt2' in str.lower(str(type(tokenizer))):
        tokenizer.pad_token = tokenizer.eos_token 


    if not single_prefix is None:
        prompt = [single_prefix]
    elif not prefixes_path is None:
        if ".txt" in prefixes_path:
            prompt = open(prefixes_path, "r").readlines()
            prompt = [p.replace("\n", "") for p in prompt]
        elif ".csv" in prefixes_path:
            prompt = pd.read_csv(prefixes_path).values[:,1].tolist()
    input_ids = tokenizer.batch_encode_plus(prompt, padding=True, truncation=True, return_tensors="pt")['input_ids'].to(device)

    # neucleus sampling (sampling=True):
    # greedy search (sampling=False): 
    generations = []
    for ids in tqdm.tqdm(input_ids):
        ids = ids[ids != tokenizer.pad_token_id]
        ids = torch.unsqueeze(ids, 0)
        generation = model.generate(ids, do_sample=sampling, max_new_tokens=generation_length, top_k=None, top_p=top_p, num_return_sequences=generations_per_prefix).tolist()
        generations += generation
    generated_texts = tokenizer.batch_decode(generations)


    if print_texts:
        for t in generated_texts:
            print(t)

    if save_texts_loc is None:
        (pd.DataFrame(generated_texts)).to_csv(save_texts_loc)

