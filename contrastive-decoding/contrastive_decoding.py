import torch
import pandas as pd
from model_comparison_helpers import instantiate_models, get_input_ids
import tqdm
from transformers import BitsAndBytesConfig
import torch


class ContrastiveDecoder:
    def __init__(self, **kwargs):
        # Default values
        self.model_name = "gpt2-xl"
        self.generation_length = 20
        self.generations_per_prefix = 1
        self.starting_model_path = "gpt2-xl"
        self.comparison_model_path = "gpt2-xl"
        self.starting_model_weight = -1
        self.comparison_model_weight = 1
        self.tokenizer_family = "gpt2"
        self.single_prefix = None
        self.prefixes_path = None
        self.set_prefix_len = 7
        self.n_prefixes = None
        self.device = "cuda:0"
        self.save_texts_loc = None
        self.text_set = None
        self.temp_save_model_loc = "/tmp/temp_"
        self.print_texts = True
        self.sampling = True
        self.num_beams = None
        self.num_beam_groups = None
        self.diversity_penalty = 0.0
        self.top_p = 0.95
        self.limit_to_starting_model_top_p = -1
        self.similarity_gating_intensity = -1
        self.comparison_model_prefix_ids = None
        self.starting_model_prefix_ids = None
        self.return_divergences = False
        self.sort_by_divergences = True
        self.return_perplexities = False
        self.include_prefix_in_divergences = True
        self.beam_search_sort = None
        self.quantize=True
        self.no_quantize_base_model=False
        self.divergence_fnct="l1"
        self.model = None
        self.starting_model = None
        self.comparison_model = None
        self.tokenizer = None
        self.cache_attn = False
        self.batch_size = 1
        self.return_all_token_divergences = False

         # Update with any arguments passed to the constructor
        for key, value in kwargs.items():
            setattr(self, key, value)

        #torch.autograd.set_detect_anomaly(True)
        with torch.no_grad():
            if self.model is None or self.tokenizer is None:
                print("Building models/tokenizer:")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
                if not self.quantize:
                    bnb_config = None
                self.model, self.starting_model, self.comparison_model, self.tokenizer = instantiate_models(
                    model_name = self.model_name,
                    starting_model_path = self.starting_model_path,
                    comparison_model_path = self.comparison_model_path,
                    starting_model_weight = self.starting_model_weight,
                    comparison_model_weight = self.comparison_model_weight,
                    tokenizer_family = self.tokenizer_family,
                    device = self.device,
                    temp_save_model_loc = self.temp_save_model_loc,
                    limit_to_starting_model_top_p = self.limit_to_starting_model_top_p,
                    similarity_gating_intensity = self.similarity_gating_intensity,
                    comparison_model_prefix_ids = self.comparison_model_prefix_ids,
                    starting_model_prefix_ids = self.starting_model_prefix_ids,
                    bnb_config = bnb_config,
                    use_8_bit = self.quantize,
                    no_quantize_base_model = self.no_quantize_base_model,
                    cache_attn = self.cache_attn
                )
    def decode(self, **kwargs):
        if kwargs:
            if self.model is None or self.tokenizer is None or "starting_model" in kwargs or "comparison_model" in kwargs:
                self.__init__(**kwargs)
            else:
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        input_ids = get_input_ids(
                self.tokenizer,
                single_prefix = self.single_prefix,
                text_set = self.text_set,
                prefixes_path = self.prefixes_path,
                set_prefix_len = self.set_prefix_len,
                n_prefixes = self.n_prefixes,
                device = self.device
            )
        if self.set_prefix_len is None:
            self.set_prefix_len = input_ids.size()[1]
        #print(input_ids)
        marker_id = self.tokenizer.convert_tokens_to_ids("|")

        generations = self.decode_loop(input_ids, 
                                  self.model, 
                                  self.tokenizer,
                                  self.generation_length, 
                                  self.sampling,
                                  self.num_beams,
                                  self.num_beam_groups,
                                  self.diversity_penalty, 
                                  self.top_p, 
                                  self.generations_per_prefix,
                                  self.batch_size
                                  )

        if self.return_divergences:
            text_ids = torch.tensor(generations).to(self.device)
            div_output = self.model.calculate_current_divergence(text_ids, 
                                                            metric = self.divergence_fnct, 
                                                            batch_size = 8,
                                                            end_tokens_to_only_consider = 0 if self.include_prefix_in_divergences else self.generation_length,
                                                            return_perplexities = self.return_perplexities,
                                                            return_all_token_divergences = self.return_all_token_divergences
                                                            )                
            divergences = torch.tensor(div_output['divergences'])
            if self.return_perplexities:
                starting_model_perplexities = div_output['starting_model_perplexities']
                comparison_model_perplexities = div_output['comparison_model_perplexities']
            if self.return_all_token_divergences:
                all_token_divergences = torch.tensor(div_output['all_token_divergences'])
                print("all_token_divergences.size", all_token_divergences.size())
                if self.print_texts:
                    # Linearly map all_token_divergences to [0, 1]
                    max_divergence = torch.max(all_token_divergences[:, self.set_prefix_len:]).item()
                    min_divergence = torch.min(all_token_divergences[:, self.set_prefix_len:]).item()
                    all_token_colorings = (all_token_divergences - min_divergence) / max(max_divergence - min_divergence, 1e-7)                    


            text_ids = torch.cat((text_ids[:,:self.set_prefix_len], torch.full((text_ids.size(0), 1), marker_id, device=self.device), text_ids[:,self.set_prefix_len:]), dim=1)
            generations = text_ids.tolist()
            generated_texts = self.tokenizer.batch_decode(generations)
            generated_tokens = [[self.tokenizer.convert_ids_to_tokens(t) for t in g] for g in generations]
            n_divergences = len(divergences)

            block_avg_divergences = []
            for j in range(0, n_divergences, self.generations_per_prefix):
                block_mean = torch.mean(divergences[j:j+self.generations_per_prefix])
                for _ in range(self.generations_per_prefix):
                    block_avg_divergences.append(block_mean)
            block_avg_divergences = torch.stack(block_avg_divergences).to(torch.float64)

            if self.beam_search_sort is None and not self.sort_by_divergences and not self.num_beams is None:
                self.beam_search_sort = True

            if self.sort_by_divergences:
                if self.single_prefix is None:
                    # Create a new divergences list whose entries are averages of the entries in each sequential span of generations_per_prefix sized blocks in decoder_divergences
                    # Add a small amount of the original divergences so texts are also sorted within blocks.
                    block_avg_divergences = block_avg_divergences + 1e-7 * divergences.to(torch.float64)
                    #print(block_avg_divergences)
                    _, indices = torch.topk(block_avg_divergences, k = n_divergences)
                else:
                    _, indices = torch.topk(divergences, k = n_divergences)
            elif self.beam_search_sort:
                print("Sorting by beam search")
                # Create indices for the original text ordering
                original_indices = torch.tensor(range(len(generated_texts)))

                # Sort original_indices by block_avg_divergences
                _, indices = torch.topk(block_avg_divergences, k = n_divergences)
                original_indices = original_indices[indices]

                # Now within each generations_per_prefix block of original_indices, sort alphabetically
                # according to the generated_texts
                for i in range(0, len(original_indices), self.generations_per_prefix):
                    block_indices = original_indices[i:i+self.generations_per_prefix]
                    block_texts = [generated_texts[j] for j in block_indices]
                    sorted_block_indices = torch.tensor([x for _, x in sorted(zip(block_texts, block_indices))])
                    original_indices[i:i+self.generations_per_prefix] = sorted_block_indices
                indices = original_indices

            text_ids = text_ids[indices]
            divergences = divergences[indices]
            generated_texts = [generated_texts[i] for i in indices]
            generated_tokens = [generated_tokens[i] for i in indices]
            generations = [generations[i] for i in indices]
            if self.return_perplexities:
                starting_model_perplexities = [starting_model_perplexities[i] for i in indices]
                comparison_model_perplexities = [comparison_model_perplexities[i] for i in indices]
            if self.return_all_token_divergences:
                all_token_divergences = all_token_divergences[indices]

                

        if self.print_texts:
            printstr = ''
            if self.return_divergences:
                for i in range(len(generated_tokens)):
                    if self.return_all_token_divergences:
                        #print(all_token_colorings.tolist())
                        # Set printed generated_texts backgrounds to be more or less red based on all_token_colorings
                        divergence_str = f"{divergences[i].item():.5f}, "
                        prompt_text = self.tokenizer.decode(generations[i][:self.set_prefix_len+1])
                        colored_text = ""
                        for j in range(self.set_prefix_len, len(all_token_colorings[i])):
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

        if not self.save_texts_loc is None:
            if self.return_divergences:
                df_list = [[d.item(),g] for g,d in zip(generated_texts, divergences)]
                cols = ["divergence", "decoding"]
            else:
                df_list = [[g] for g in generated_texts]
                cols = ["decoding"]
            if self.return_perplexities:
                df_list = [item + [smp,cmp] for item,smp,cmp in zip(df_list, starting_model_perplexities, comparison_model_perplexities)]
                cols = cols + ["starting_model_perplexity", "comparison_model_perplexity"]
            if self.return_all_token_divergences:
                df_list = [item + [per_token_divs] for item,per_token_divs in zip(df_list,all_token_divergences.tolist())]
                cols = cols + ["all_token_divergences"]
            df = pd.DataFrame(data = df_list, columns = cols)
            df.to_csv(self.save_texts_loc)
        
        result = {"texts": generated_texts}
        if self.return_divergences:
            result["divergences"] = divergences
        if self.return_perplexities:
            result["starting_model_perplexities"] = starting_model_perplexities
            result["comparison_model_perplexities"] = comparison_model_perplexities
        if self.return_all_token_divergences:
            result["all_token_divergences"] = all_token_divergences.tolist()
        return result

    def decode_loop(self, 
                    input_ids, 
                    model, 
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
                                                temperature=temperature,
                                                return_dict_in_generate=True
                                                ).sequences.tolist()
            else:
                generations_batch = model.generate(batch_ids, 
                                                do_sample=sampling, 
                                                max_new_tokens=generation_length, 
                                                min_length=output_len, 
                                                top_k=None, 
                                                top_p=top_p, 
                                                num_return_sequences=generations_per_prefix,
                                                temperature=temperature,
                                                return_dict_in_generate=True
                                                ).sequences.tolist()
            generations += generations_batch
        return generations