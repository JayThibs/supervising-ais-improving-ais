import os
import pickle
import torch
import pandas as pd
from tqdm import tqdm
from transformers import BitsAndBytesConfig, PreTrainedTokenizer
from typing import Optional, List
import datetime

from model_comparison_helpers import instantiate_models, get_input_ids


class ContrastiveDecoder:
    def __init__(self, **kwargs):
        # Default values
        self.model_name : str = "gpt2-xl"
        self.generation_length : int = 20
        self.generations_per_prefix : int = 1
        self.starting_model_path : str = "gpt2-xl"
        self.comparison_model_path : str = "gpt2-xl"
        self.starting_model_weight : float = -1
        self.comparison_model_weight : float = 1
        self.tokenizer_family : str = "gpt2"
        self.single_prefix : Optional[str] = None
        self.prefixes_path : Optional[str] = None
        self.set_prefix_len : int = 7
        self.n_prefixes : Optional[int] = None
        self.device : str = "cuda:0"
        self.starting_model_device : Optional[str] = None
        self.comparison_model_device : Optional[str] = None
        self.save_texts_loc : Optional[str] = None
        self.text_set : Optional[str] = None
        self.print_texts : bool = True
        self.sampling : bool = True
        self.num_beams : Optional[int] = None
        self.num_beam_groups : Optional[int] = None
        self.diversity_penalty : float = 0.0
        self.top_p : float = 0.95
        self.limit_to_starting_model_top_p : Optional[float] = None
        self.similarity_gating_intensity : Optional[float] = None
        self.comparison_model_prefix_ids : Optional[List[int]] = None
        self.starting_model_prefix_ids : Optional[List[int]] = None
        self.return_divergences : bool = False
        self.return_prefix_divergences : bool = False
        self.sort_by_divergences : bool = True
        self.return_perplexities : bool = False
        self.include_prefix_in_divergences : bool = True
        self.beam_search_sort : Optional[bool] = None
        self.quantize : bool = True
        self.no_quantize_starting_model : bool = False
        self.use_avg_KL_as_divergences : bool = True
        self.model : Optional[torch.nn.Module] = None
        self.comparison_model : Optional[torch.nn.Module] = None
        self.tokenizer : Optional[PreTrainedTokenizer] = None
        self.cache_attn : bool = False
        self.batch_size : int = 1
        self.return_all_token_divergences : bool = False
        self.comparison_model_interpolation_weight : float = 0.0
        self.save_interval : Optional[int] = None
        self.generations = []
        self.temperature : float = 0.6
        self.divergences = []
        self.starting_model_perplexities = []
        self.comparison_model_perplexities = []
        self.all_token_divergences = []
        
        # Update with any arguments passed to the constructor
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.device = self._get_device(kwargs.get('device', 'auto'))
        
        #torch.autograd.set_detect_anomaly(True)
        with torch.no_grad():
            if self.model is None or self.tokenizer is None:
                print("Building models/tokenizer:")
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_use_double_quant=True,
                    bnb_8bit_quant_type="nf8",
                    bnb_8bit_compute_dtype=torch.bfloat16
                )
                if not self.quantize:
                    bnb_config = None
                self.model, self.comparison_model, self.tokenizer = instantiate_models(
                    model_name = self.model_name,
                    starting_model_path = self.starting_model_path,
                    comparison_model_path = self.comparison_model_path,
                    starting_model_weight = self.starting_model_weight,
                    comparison_model_weight = self.comparison_model_weight,
                    tokenizer_family = self.tokenizer_family,
                    device = self.device,
                    starting_model_device = self.starting_model_device,
                    comparison_model_device = self.comparison_model_device,
                    limit_to_starting_model_top_p = self.limit_to_starting_model_top_p,
                    similarity_gating_intensity = self.similarity_gating_intensity,
                    comparison_model_prefix_ids = self.comparison_model_prefix_ids,
                    starting_model_prefix_ids = self.starting_model_prefix_ids,
                    bnb_config = bnb_config,
                    no_quantize_starting_model = self.no_quantize_starting_model,
                    cache_attn = self.cache_attn,
                    comparison_model_interpolation_weight = self.comparison_model_interpolation_weight
                )

        # Ensure self.model is the custom ContrastiveLM instance
        if not hasattr(self.model, 'calculate_current_divergence'):
            raise AttributeError("The model does not have the 'calculate_current_divergence' method. Make sure you're using the custom ContrastiveLM model.")

    def _get_device(self, device):
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)

    def decode(self, **kwargs) -> dict:
        if kwargs:
            if self.model is None or self.tokenizer is None or "starting_model" in kwargs or "comparison_model" in kwargs:
                self.__init__(**kwargs)
            else:
                for key, value in kwargs.items():
                    setattr(self, key, value)
        self.tokenizer.padding_side = "left"
        #print("text_set", self.text_set)
        #print(self.tokenizer,
        #      self.single_prefix,
        #      self.text_set,
        #      self.prefixes_path,
        #      self.set_prefix_len,
        #      self.n_prefixes,
        #      self.device)
        
        input_ids = get_input_ids(
                self.tokenizer,
                single_prefix = self.single_prefix,
                text_set = self.text_set,
                prefixes_path = self.prefixes_path,
                set_prefix_len = self.set_prefix_len,
                n_prefixes = self.n_prefixes,
                device = self.device
            )
        #print("input_tokens", self.tokenizer.decode(input_ids[0]))
        if self.set_prefix_len is None:
            self.set_prefix_len = input_ids.size()[1]
        marker_id = self.tokenizer.convert_tokens_to_ids("|")
        #print(input_ids)

        # Ensure we're only using the specified number of examples
        if self.n_prefixes is not None:
            input_ids = input_ids[:self.n_prefixes]

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
                                  self.batch_size,
                                  self.temperature
                                  )

        print(f"Number of generations: {len(generations)}")


        if self.return_divergences:
            with torch.no_grad():
                divergences = []
                all_token_divergences = []
                starting_model_perplexities = []
                comparison_model_perplexities = []

                for i in range(0, len(self.generations), self.batch_size):
                    batch_generations = self.generations[i:i+self.batch_size]
                    text_ids = torch.tensor(batch_generations).to(self.device)
                    div_output = self.model.calculate_current_divergence(text_ids, 
                                                                         batch_size = self.batch_size,
                                                                         end_tokens_to_only_consider = 0 if self.include_prefix_in_divergences else self.generation_length,
                                                                         return_perplexities = self.return_perplexities,
                                                                         return_all_token_divergences = self.return_all_token_divergences,
                                                                         use_avg_KL_as_divergences = self.use_avg_KL_as_divergences
                                                                    )                
                    divergences.extend(div_output['divergences'])
                    if self.return_perplexities:
                        starting_model_perplexities.extend(div_output['starting_model_perplexities'])
                        comparison_model_perplexities.extend(div_output['comparison_model_perplexities'])
                    if self.return_all_token_divergences:
                        all_token_divergences.extend(div_output['all_token_divergences'])

                divergences = torch.tensor(divergences)
                if self.return_all_token_divergences:
                    all_token_divergences = torch.tensor(all_token_divergences)
                    print("all_token_divergences.size", all_token_divergences.size())

                print(f"After divergence calculation - Number of divergences: {len(divergences)}")

                text_ids = torch.tensor(self.generations).to(self.device)
                text_ids = torch.cat((text_ids[:,:self.set_prefix_len], torch.full((text_ids.size(0), 1), marker_id, device=self.device), text_ids[:,self.set_prefix_len:]), dim=1)
                generations = text_ids.tolist()
                generated_texts = self.tokenizer.batch_decode(generations)
                generated_tokens = [[self.tokenizer.convert_ids_to_tokens(t) for t in g] for g in generations]
                n_divergences = len(divergences)

                print(f"After text decoding - Number of texts: {len(generated_texts)}")

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
                    # Sorting by beam search
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
                else:
                    # No sorting
                    indices = torch.tensor(range(len(generated_texts)))

                # Ensure all lists have the same length
                min_length = min(len(generated_texts), len(divergences), len(self.generations))
                generated_texts = generated_texts[:min_length]
                divergences = divergences[:min_length]
                self.generations = self.generations[:min_length]

                print(f"After length adjustment - Number of texts: {len(generated_texts)}")

                if self.return_perplexities:
                    starting_model_perplexities = starting_model_perplexities[:min_length]
                    comparison_model_perplexities = comparison_model_perplexities[:min_length]
                if self.return_all_token_divergences:
                    all_token_divergences = all_token_divergences[:min_length]

                # Sort by divergence
                indices = list(range(min_length))
                if self.sort_by_divergences:
                    indices = sorted(indices, key=lambda i: divergences[i], reverse=True)

                generated_texts = [generated_texts[i] for i in indices]
                divergences = [divergences[i] for i in indices]
                self.generations = [self.generations[i] for i in indices]
                if self.return_perplexities:
                    starting_model_perplexities = [starting_model_perplexities[i] for i in indices]
                    comparison_model_perplexities = [comparison_model_perplexities[i] for i in indices]
                if self.return_all_token_divergences:
                    all_token_divergences = all_token_divergences[indices]

                self.divergences = divergences

                print(f"After sorting - Number of texts: {len(generated_texts)}")

        if self.print_texts:
            printstr = ''
            if self.return_divergences:
                BOS_token = self.tokenizer.bos_token
                space_token = self.tokenizer.tokenize(" ")[0]
                newline_token = self.tokenizer.tokenize("\n")[0]
                
                for i in range(len(generated_texts)):
                    if self.return_all_token_divergences:
                        divergence_str = f"{divergences[i]:.5f}, "
                        prompt_text = self.tokenizer.decode(generations[i][:self.set_prefix_len+1])
                        colored_text = ""
                        for j in range(self.set_prefix_len, len(all_token_divergences[i])):
                            token_color = all_token_divergences[i][j].item()
                            token_text = generated_tokens[i][j+1]
                            token_text = token_text.replace(BOS_token, "")
                            token_text = token_text.replace(space_token, " ")
                            token_text = token_text.replace(newline_token, "\\n")
                            colored_text += f"\033[48;2;{int(255*token_color)};0;0m{token_text}\033[48;2;0;0;0m"
                        printstr += divergence_str + prompt_text + colored_text
                    else:
                        printstr += str(round(divergences[i], 5)) + ", " + generated_texts[i]
                    if i < len(generated_texts) - 1:
                        printstr += '\033[48;2;0;0;0m\n'
            else:
                for i in range(len(generated_texts)):
                    printstr += generated_texts[i]
                    if i < len(generated_texts) - 1:
                        printstr += '\n'
            print(printstr)

        result = {"texts": generated_texts}
        if self.return_divergences:
            result["divergences"] = divergences
        if self.return_prefix_divergences:
            prefix_divergences = []
            for j in range(0, n_divergences, self.generations_per_prefix):
                block_mean = torch.mean(divergences[j:j+self.generations_per_prefix])
                prefix_divergences.append(block_mean)
            result["prefix_divergences"] = torch.tensor(prefix_divergences)
        if self.return_perplexities:
            result["starting_model_perplexities"] = starting_model_perplexities
            result["comparison_model_perplexities"] = comparison_model_perplexities
        if self.return_all_token_divergences:
            result["all_token_divergences"] = all_token_divergences.tolist()

        print("Result lengths:")
        for key, value in result.items():
            print(f"{key}: {len(value)}")

        # Save all results
        self.save_all_results(result)

        return result

    def decode_loop(self, 
                    input_ids : torch.Tensor, 
                    model : torch.nn.Module, 
                    tokenizer : PreTrainedTokenizer, 
                    generation_length : int = 20, 
                    sampling : bool = True, 
                    num_beams : Optional[int] = None,
                    num_beam_groups : Optional[int] = None,
                    diversity_penalty : float = 0.0, 
                    top_p : float = 0.95, 
                    generations_per_prefix : int = 1, 
                    batch_size : int = 1,
                    temperature : float = 0.6
                    ) -> List[List[int]]:
        with torch.no_grad():
            self.generations = []
            self.divergences = []  # Initialize divergences list
            n_inputs = input_ids.size()[0]
            output_len = input_ids.size()[1] + generation_length
            for i in tqdm(range(0, n_inputs, batch_size)):
                batch_ids = input_ids[i:min(i+batch_size, n_inputs)]

                # Input could be left padded, so we need to create an attention mask
                attention_mask = torch.ones_like(batch_ids)
                attention_mask[batch_ids == self.tokenizer.pad_token_id] = 0
                
                try:
                    if not num_beams is None:
                        generations_batch = model.generate(batch_ids, 
                                                           attention_mask=attention_mask,
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
                except NotImplementedError as e:
                    if "MPS" in str(e):
                        print("MPS device not supported for this operation. Falling back to CPU.")
                        model = model.to("cpu")
                        batch_ids = batch_ids.to("cpu")
                        attention_mask = attention_mask.to("cpu")
                        if not num_beams is None:
                            generations_batch = model.generate(batch_ids, 
                                                               attention_mask=attention_mask,
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
                        model = model.to(self.device)
                    else:
                        raise e
                
                self.generations += generations_batch
                
                # If divergences are calculated, update self.divergences
                if self.return_divergences:
                    with torch.no_grad():
                        div_output = self.model.calculate_current_divergence(
                            torch.tensor(generations_batch).to(self.device),
                            batch_size=self.batch_size,
                            end_tokens_to_only_consider=0 if self.include_prefix_in_divergences else self.generation_length,
                            return_perplexities=self.return_perplexities,
                            return_all_token_divergences=self.return_all_token_divergences,
                            use_avg_KL_as_divergences=self.use_avg_KL_as_divergences
                        )
                        self.divergences.extend(div_output['divergences'])

                # Save results periodically
                if self.save_interval and (i + batch_size) % self.save_interval == 0:
                    self.save_partial_results(i + batch_size)

                # Print generated text
                if self.print_texts:
                    self.print_generated_texts(i, min(i + batch_size, n_inputs))

        print(f"Total number of generations: {len(self.generations)}")
        return self.generations
    
    def save_all_results(self, result):
        if self.save_texts_loc is not None:
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(self.save_texts_loc), exist_ok=True)

            # Generate a timestamp for the current run
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            # Print result keys and lengths
            print("Result keys:", result.keys())
            for key, value in result.items():
                print(f"{key} length: {len(value) if isinstance(value, (list, dict)) else 1}")

            # Save as CSV with timestamp
            df = pd.DataFrame(result)
            csv_filename = f"{os.path.splitext(os.path.basename(self.save_texts_loc))[0]}_{timestamp}_all_results.csv"
            csv_path = os.path.join('outputs', csv_filename)
            df.to_csv(csv_path, index=False)
            print(f"Saved all results to {csv_path}")
            print(f"Number of rows in CSV: {len(df)}")

            # Save as pickle for more complex data structures with timestamp
            pickle_filename = f"{os.path.splitext(os.path.basename(self.save_texts_loc))[0]}_{timestamp}_all_results.pkl"
            pickle_path = os.path.join('outputs', pickle_filename)
            with open(pickle_path, 'wb') as f:
                pickle.dump(result, f)
            print(f"Saved all results (including complex data structures) to {pickle_path}")
    
    def save_partial_results(self, num_examples):
        partial_results = {
            "generations": self.generations[:num_examples],
            "divergences": self.divergences[:num_examples] if self.divergences else None,
            "starting_model_perplexities": self.starting_model_perplexities[:num_examples] if self.starting_model_perplexities is not None else None,
            "comparison_model_perplexities": self.comparison_model_perplexities[:num_examples] if self.comparison_model_perplexities is not None else None,
            "all_token_divergences": self.all_token_divergences[:num_examples] if hasattr(self, 'all_token_divergences') else None,
        }

        # Split the original save path into directory, filename, and extension
        save_dir, save_filename = os.path.split(self.save_texts_loc)
        save_name, _ = os.path.splitext(save_filename)

        # Construct the new partial save path
        partial_save_filename = f"{save_name}_partial_{num_examples}.pkl"
        partial_save_path = os.path.join(save_dir, partial_save_filename)

        with open(partial_save_path, "wb") as f:
            pickle.dump(partial_results, f)

        print(f"Saved partial results for {num_examples} examples to {partial_save_path}")

    def print_generated_texts(self, start_idx, end_idx):
        for idx in range(start_idx, min(end_idx, len(self.generations))):
            generated_text = self.tokenizer.decode(self.generations[idx])
            print(f"Example {idx + 1}:")
            print(generated_text)
            print("-" * 50)