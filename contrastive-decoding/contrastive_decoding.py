import torch
import pandas as pd
from model_comparison_helpers import instantiate_models, get_input_ids
import tqdm
from transformers import BitsAndBytesConfig
import gc
import torch
import transformers
#from transformers import AutoModel, AutoModelForCausalLM, GPT2LMHeadModel, OPTForCausalLM, AutoTokenizer
#import os
#from model_comparison_helpers import build_contrastive_lm, get_cls, instantiate_models
#import numpy as np
#import shutil


def decode_loop(input_ids, 
                model, 
                starting_model, 
                comparison_model, 
                tokenizer, 
                generation_length, 
                sampling, 
                num_beams,
                num_beam_groups,
                diversity_penalty, 
                top_p, 
                generations_per_prefix, 
                batch_size = 1,
                temperature = 0.6):
    generations = []
    n_inputs = input_ids.size()[0]
    output_len = input_ids.size()[1] + generation_length
    for i in tqdm.tqdm(range(0, n_inputs, batch_size)):
        batch_ids = input_ids[i:min(i+batch_size, n_inputs)]

        n_pads = torch.sum(batch_ids == tokenizer.pad_token_id)
        if n_pads > 0 and not "gpt" in str(type(tokenizer)).lower():
            raise ValueError("input_ids should not contain any pad tokens")
        if not num_beams is None:
            generations_batch = model.generate(batch_ids, 
                                               do_sample=sampling, 
                                               max_new_tokens=generation_length, 
                                               min_length=output_len, 
                                               top_k=None, 
                                               top_p=top_p, 
                                               num_return_sequences=generations_per_prefix,
                                               num_beams=num_beams,
                                               num_beam_groups=num_beam_groups,
                                               diversity_penalty=diversity_penalty,
                                               temperature=temperature
                                               ).tolist()
        else:
            generations_batch = model.generate(batch_ids, 
                                               do_sample=sampling, 
                                               max_new_tokens=generation_length, 
                                               min_length=output_len, 
                                               top_k=None, 
                                               top_p=top_p, 
                                               num_return_sequences=generations_per_prefix,
                                               temperature=temperature
                                               ).tolist()
        generations += generations_batch
    return generations


def decode(
        model_name = "gpt2-xl",
        generation_length = 20,
        generations_per_prefix = 1,
        starting_model_path = "gpt2-xl",
        comparison_model_path = "gpt2-xl",
        starting_model_weight = -1,
        comparison_model_weight = 1,
        tokenizer_family = "gpt2",
        single_prefix = None,
        prefixes_path = None,
        set_prefix_len = 7,
        n_prefixes = None,
        device = "cuda:0",
        save_texts_loc = None,
        text_set = None,
        temp_save_model_loc = "/tmp/temp_",
        print_texts = True,
        sampling = True,
        num_beams = None,
        num_beam_groups = None,
        diversity_penalty = 0.0, 
        top_p = 0.95,
        limit_to_starting_model_top_p = -1,
        similarity_gating_intensity = -1,
        comparison_model_prefix_ids = None,
        starting_model_prefix_ids = None,
        return_divergences = False,
        sort_by_divergences = True,
        return_perplexities = False,
        include_prefix_in_divergences = True,
        beam_search_sort = None,
        quantize=True,
        no_quantize_base_model=False,
        divergence_fnct="l1",
        model = None,
        starting_model = None,
        comparison_model = None,
        tokenizer = None,
        cache_attn = False,
        batch_size = 1,
        return_all_token_divergences = False
    ): 
    #torch.autograd.set_detect_anomaly(True)
    with torch.no_grad():
        if model is None or tokenizer is None:
            print("Building models/tokenizer:")
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
                bnb_config = bnb_config,
                use_8_bit=quantize,
                no_quantize_base_model=no_quantize_base_model,
                cache_attn = cache_attn
            )
            # Find any python objects that are also huggingface models
            # huggingface_models = [obj for obj in locals().values() if isinstance(obj, transformers.PreTrainedModel)]
            # for e in huggingface_models:
            #     print('\n\n\n\n', e, '\n\n\n\n')
        
        
        input_ids = get_input_ids(
            tokenizer,
            single_prefix = single_prefix,
            text_set = text_set,
            prefixes_path = prefixes_path,
            set_prefix_len = set_prefix_len,
            n_prefixes = n_prefixes,
            device = device
            )
        if set_prefix_len is None:
            set_prefix_len = input_ids.size()[1]
        #print(input_ids)
        marker_id = tokenizer.convert_tokens_to_ids("|")

        generations = decode_loop(input_ids, 
                                  model, 
                                  starting_model, 
                                  comparison_model, 
                                  tokenizer,
                                  generation_length, 
                                  sampling,
                                  num_beams,
                                  num_beam_groups,
                                  diversity_penalty, 
                                  top_p, 
                                  generations_per_prefix,
                                  batch_size
                                  )

        if return_divergences:
            text_ids = torch.tensor(generations).to(device)
            div_output = model.calculate_current_divergence(text_ids, 
                                                            metric = divergence_fnct, 
                                                            batch_size = 8,
                                                            end_tokens_to_only_consider = 0 if include_prefix_in_divergences else generation_length,
                                                            return_perplexities = return_perplexities,
                                                            return_all_token_divergences = return_all_token_divergences
                                                            )                
            divergences = torch.tensor(div_output['divergences'])
            if return_perplexities:
                starting_model_perplexities = div_output['starting_model_perplexities']
                comparison_model_perplexities = div_output['comparison_model_perplexities']
            if return_all_token_divergences:
                all_token_divergences = torch.tensor(div_output['all_token_divergences'])
                print("all_token_divergences.size", all_token_divergences.size())
                if print_texts:
                    # Linearly map all_token_divergences to [0, 1]
                    max_divergence = torch.max(all_token_divergences[:, set_prefix_len:]).item()
                    min_divergence = torch.min(all_token_divergences[:, set_prefix_len:]).item()
                    all_token_colorings = (all_token_divergences - min_divergence) / max(max_divergence - min_divergence, 1e-7)                    


            text_ids = torch.cat((text_ids[:,:set_prefix_len], torch.full((text_ids.size(0), 1), marker_id, device=device), text_ids[:,set_prefix_len:]), dim=1)
            generations = text_ids.tolist()
            generated_texts = tokenizer.batch_decode(generations)
            generated_tokens = [[tokenizer.convert_ids_to_tokens(t) for t in g] for g in generations]
            n_divergences = len(divergences)

            block_avg_divergences = []
            for j in range(0, n_divergences, generations_per_prefix):
                block_mean = torch.mean(divergences[j:j+generations_per_prefix])
                for _ in range(generations_per_prefix):
                    block_avg_divergences.append(block_mean)
            block_avg_divergences = torch.stack(block_avg_divergences).to(torch.float64)

            if beam_search_sort is None and not sort_by_divergences and not num_beams is None:
                beam_search_sort = True

            if sort_by_divergences:
                if single_prefix is None:
                    # Create a new divergences list whose entries are averages of the entries in each sequential span of generations_per_prefix sized blocks in decoder_divergences
                    # Add a small amount of the original divergences so texts are also sorted within blocks.
                    block_avg_divergences = block_avg_divergences + 1e-7 * divergences.to(torch.float64)
                    #print(block_avg_divergences)
                    _, indices = torch.topk(block_avg_divergences, k = n_divergences)
                else:
                    _, indices = torch.topk(divergences, k = n_divergences)
            elif beam_search_sort:
                print("Sorting by beam search")
                # Create indices for the original text ordering
                original_indices = torch.tensor(range(len(generated_texts)))

                # Sort original_indices by block_avg_divergences
                _, indices = torch.topk(block_avg_divergences, k = n_divergences)
                original_indices = original_indices[indices]

                # Now within each generations_per_prefix block of original_indices, sort alphabetically
                # according to the generated_texts
                for i in range(0, len(original_indices), generations_per_prefix):
                    block_indices = original_indices[i:i+generations_per_prefix]
                    block_texts = [generated_texts[j] for j in block_indices]
                    sorted_block_indices = torch.tensor([x for _, x in sorted(zip(block_texts, block_indices))])
                    original_indices[i:i+generations_per_prefix] = sorted_block_indices
                indices = original_indices

            text_ids = text_ids[indices]
            divergences = divergences[indices]
            generated_texts = [generated_texts[i] for i in indices]
            generated_tokens = [generated_tokens[i] for i in indices]
            generations = [generations[i] for i in indices]
            if return_perplexities:
                starting_model_perplexities = [starting_model_perplexities[i] for i in indices]
                comparison_model_perplexities = [comparison_model_perplexities[i] for i in indices]
            if return_all_token_divergences:
                all_token_divergences = all_token_divergences[indices]

                

        if print_texts:
            printstr = ''
            if return_divergences:
                for i in range(len(generated_tokens)):
                    if return_all_token_divergences:
                        #print(all_token_colorings.tolist())
                        # Set printed generated_texts backgrounds to be more or less red based on all_token_colorings
                        divergence_str = f"{divergences[i].item():.5f}, "
                        prompt_text = tokenizer.decode(generations[i][:set_prefix_len+1])
                        colored_text = ""
                        for j in range(set_prefix_len, len(all_token_colorings[i])):
                            token_color = all_token_colorings[i][j].item()
                            colored_text += f"\033[48;2;{int(255*token_color)};0;0m{generated_tokens[i][j+1]}\033[48;2;0;0;0m"
                        printstr += divergence_str + prompt_text + colored_text
                    else:
                        printstr += str(round(divergences[i].item(), 5)) + ", " + generated_texts[i]
                    if i < len(generated_texts) - 1:
                        printstr += '\033[48;2;0;0;0m\n'
            else:
                for i in range(len(generated_texts)):
                    printstr += generated_texts[i]
                    if i < len(generated_texts) - 1:
                        printstr += '\n'
            print(printstr)

        if not save_texts_loc is None:
            if return_divergences:
                df_list = [[d.item(),g] for g,d in zip(generated_texts, divergences)]
                cols = ["divergence", "decoding"]
            else:
                df_list = [[g] for g in generated_texts]
                cols = ["decoding"]
            if return_perplexities:
                df_list = [item + [smp,cmp] for item,smp,cmp in zip(df_list, starting_model_perplexities, comparison_model_perplexities)]
                cols = cols + ["starting_model_perplexity", "comparison_model_perplexity"]
            if return_all_token_divergences:
                df_list = [item + [per_token_divs] for item,per_token_divs in zip(df_list,all_token_divergences.tolist())]
                cols = cols + ["all_token_divergences"]
            df = pd.DataFrame(data = df_list, columns = cols)
            df.to_csv(save_texts_loc)
        
        result = {"texts": generated_texts}
        if return_divergences:
            result["divergences"] = divergences
        if return_perplexities:
            result["starting_model_perplexities"] = starting_model_perplexities
            result["comparison_model_perplexities"] = comparison_model_perplexities
        if return_all_token_divergences:
            result["all_token_divergences"] = all_token_divergences.tolist()
        return result
