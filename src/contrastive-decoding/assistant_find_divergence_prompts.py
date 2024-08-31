import torch
from model_comparison_helpers import instantiate_models
import pandas as pd
import json
from openai import OpenAI
import re
from random import sample
from contrastive_decoding import ContrastiveDecoder
import numpy as np
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, AutoModel
from terminaltables import AsciiTable
import textwrap
from typing import Optional, List, Tuple
from quick_cluster import read_past_embeddings_or_generate_new

def get_results_embeddings_and_divergences(texts : List[str],
                                           local_embedding_model_str : str,
                                           local_embedding_model : PreTrainedModel,
                                           local_tokenizer : PreTrainedTokenizer,
                                           device : str
                                           ) -> List[List[float]]:
    embeddings_list = read_past_embeddings_or_generate_new(path = None, 
                                                           client = None, 
                                                           decoded_strs = texts, 
                                                           local_embedding_model_str = local_embedding_model_str, 
                                                           local_embedding_model = local_embedding_model, 
                                                           tokenizer = local_tokenizer, 
                                                           device = device, 
                                                           recompute_embeddings = True, 
                                                           batch_size = 8, 
                                                           save_embeddings = False, 
                                                           tqdm_disable = True, 
                                                           clustering_instructions = "Identify the topic or theme of the given texts", 
                                                           max_length = 512)
    embeddings_list = torch.tensor([e.tolist() for e in embeddings_list])
    embeddings_list = embeddings_list.tolist()
    return embeddings_list
    

class DivergenceFinder:
    def __init__(self, **kwargs):
        # Default values
        self.model_name : str = "gpt2-xl"
        self.generation_length : int = 25
        self.n_cycles_ask_assistant : int = 2
        self.max_failed_cycles : int = 20
        self.openai_model_str : Optional[str] = None
        self.local_model_str : str = "NousResearch/Meta-Llama-3-8B-Instruct"
        self.generations_per_prefix : int = 5
        self.starting_model_path : str = "NousResearch/Meta-Llama-3-8B-Instruct"
        self.comparison_model_path : str = "NousResearch/Meta-Llama-3-8B"
        self.starting_model_weight : float = 1
        self.comparison_model_weight : float = -1
        self.comparison_model_interpolation_weight : float = 0.0
        self.decode_only_from_comparison_model : bool = True
        self.tokenizer_family : str = "gpt2"
        self.set_prefix_len : int = 30
        self.device : str = "cuda:0"
        self.local_model_device : str = "cuda:0"
        self.limit_to_starting_model_top_p : Optional[float] = None
        self.similarity_gating_intensity : Optional[float] = None
        self.comparison_model_prefix_ids : Optional[List[int]] = None
        self.starting_model_prefix_ids : Optional[List[int]] = None
        self.n_past_texts_subsampled : int = 10
        self.subsampling_randomness_temperature : float = 0.5
        self.api_key_path : Optional[str] = None
        self.prompts_json_path : str = "assistant_prompts/who_is_harry_potter_find_high_div_prompts.json"
        self.results_save_path : Optional[str] = None
        self.quantize : bool = True
        self.include_prefix_in_divergences : bool = False
        self.verbose : bool = True
        self.max_width : int = 65
        self.use_custom_selection_criterion : bool = True
        self.use_custom_selection_criterion_examples : bool = False
        self.use_custom_selection_criterion_for_scoring : bool = False
        self.require_custom_selection_for_inclusion : bool = False
        self.require_unique_additional_texts : bool = True
        self.use_embeddings_diversity_score : bool = False
        self.divergence_weight : float = 1.0
        self.custom_selection_criterion_weight : float = 1.0
        self.embedding_diversity_weight : float = 1.0
        self.randomize_in_context_divs : bool = False
        self.include_context_divs : bool = True
        self.local_embedding_model_str : str = "nvidia/NV-Embed-v1"
        self.n_repeat : int = 1
        self.temperature : float = 1.0

        # Update with any arguments passed to the constructor
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Define quantization config
        self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
        )
        if self.openai_model_str is None and self.local_model_str is not None:
            try:
                local_device_map = {"": int(self.local_model_device.split(":")[1])}
            except:
                local_device_map = "auto"
            self.local_model = AutoModelForCausalLM.from_pretrained(self.local_model_str,
                                                                    load_in_8bit=True, 
                                                                    device_map=local_device_map)#,
                                                                    #quantization_config=self.bnb_config)
            self.local_tokenizer = AutoTokenizer.from_pretrained(self.local_model_str)
        elif self.openai_model_str is not None and self.local_model_str is not None:
            raise ValueError("You cannot specify both openai_model_str and local_model_str.")
        elif self.openai_model_str is None and self.local_model_str is None:
            raise ValueError("You must specify either openai_model_str or local_model_str.")
        
        if not self.quantize:
            self.bnb_config = None
        self.model, self.comparison_model, self.tokenizer = instantiate_models(
            model_name = self.model_name,
            starting_model_path = self.starting_model_path,
            comparison_model_path = self.comparison_model_path,
            starting_model_weight = self.starting_model_weight,
            comparison_model_weight = self.comparison_model_weight,
            tokenizer_family = self.tokenizer_family,
            device = self.device,
            limit_to_starting_model_top_p = self.limit_to_starting_model_top_p,
            similarity_gating_intensity = self.similarity_gating_intensity,
            comparison_model_prefix_ids = self.comparison_model_prefix_ids,
            starting_model_prefix_ids = self.starting_model_prefix_ids,
            bnb_config = self.bnb_config,
        )
        if self.use_embeddings_diversity_score:
            # Load local embedding model from HuggingFace
            self.local_embedding_tokenizer = AutoTokenizer.from_pretrained(self.local_embedding_model_str)

            if self.local_embedding_model_str == "nvidia/NV-Embed-v1":
                self.local_embedding_model = AutoModel.from_pretrained(self.local_embedding_model_str, trust_remote_code = True, load_in_8bit=True, torch_dtype=torch.float16, device_map={"": 0} if self.device == "cuda:0" else "auto")
            else:
                self.local_embedding_model = AutoModel.from_pretrained(self.local_embedding_model_str).to(self.device)
        else:
            self.local_embedding_model = None
            self.local_embedding_tokenizer = None
    
        # Load system prompt, user instructions, and any seed demonstrations from prompts_json_path:
        with open(self.prompts_json_path, 'r') as f:
            self.prompts_data = json.load(f)
        
        self.system_start_prompt = self.prompts_data['system_start']
        self.system_loop_prompt = self.prompts_data['system_loop']
        self.system_end_prompt = self.prompts_data['system_end']
        self.user_start_prompt = self.prompts_data['user_start']
        self.user_loop_prompt = self.prompts_data['user_loop']
        self.user_end_prompt = self.prompts_data['user_end']
        self.seed_demonstrations_list = self.prompts_data['seed_demonstrations']
        if self.use_custom_selection_criterion:
            self.custom_selection_criterion = self.prompts_data['custom_selection_criterion']
            self.custom_selection_criterion_yes_response = self.prompts_data['custom_selection_criterion_yes_response']
            self.custom_selection_criterion_no_response = self.prompts_data['custom_selection_criterion_no_response']
        if self.use_custom_selection_criterion_examples:
            self.custom_selection_criterion_examples = self.prompts_data['custom_selection_criterion_examples']
        else:
            self.custom_selection_criterion_examples = None

        contrastive_decoder_params = {
            "model": self.model,
            "comparison_model": self.comparison_model,
            "tokenizer": self.tokenizer,
            "generation_length": self.generation_length,
            "generations_per_prefix": self.generations_per_prefix,
            "starting_model_weight": self.starting_model_weight,
            "comparison_model_weight": self.comparison_model_weight,
            "text_set": self.seed_demonstrations_list,
            "set_prefix_len": self.set_prefix_len,
            "device": self.device,
            "print_texts": False,
            "limit_to_starting_model_top_p": self.limit_to_starting_model_top_p,
            "comparison_model_prefix_ids": self.comparison_model_prefix_ids,
            "starting_model_prefix_ids": self.starting_model_prefix_ids,
            "return_divergences": True,
            "return_prefix_divergences": True,
            "sort_by_divergence": False,
            "include_prefix_in_divergences": self.include_prefix_in_divergences,
            "quantize": self.quantize,
            "comparison_model_interpolation_weight": self.comparison_model_interpolation_weight,
            "cache_attn": True
        }
        self.contrastive_decoder = ContrastiveDecoder(**contrastive_decoder_params)
        # Read OpenAI auth key from a file
        if self.api_key_path is not None:
            with open(self.api_key_path, 'r') as file:
                openai_auth_key = file.read().strip()

            # Authenticate with the OpenAI API
            self.client = OpenAI(api_key=openai_auth_key)
        else:
            self.client = None

    def find_diverging_texts(self, **kwargs) -> List[Tuple[float, str, str, int]]:
        if kwargs:
            for key, value in kwargs.items():
                setattr(self, key, value)
        all_divergences_and_texts = []
        all_full_divergence_stats = []
        all_subsample_divergence_stats = []
        all_full_diversity_stats = []
        all_subsample_diversity_stats = []
        for i in range(self.n_repeat):
            combined_loop_divergences_and_texts, combined_loop_full_divergence_stats, combined_loop_full_diversity_stats, combined_loop_subsample_divergence_stats, combined_loop_subsample_diversity_stats = self.search_loop(all_divergences_and_texts, repeat_n=i)

            all_divergences_and_texts.extend([e + (i,) for e in combined_loop_divergences_and_texts])
            all_full_divergence_stats.extend(combined_loop_full_divergence_stats)
            all_full_diversity_stats.extend(combined_loop_full_diversity_stats)
            all_subsample_divergence_stats.extend(combined_loop_subsample_divergence_stats)
            all_subsample_diversity_stats.extend(combined_loop_subsample_diversity_stats)

            print(i, "all_divergences_and_texts", all_divergences_and_texts)

        if self.results_save_path is not None:
            print("Saving results to", self.results_save_path)
            print(f"Number of examples: {len(all_divergences_and_texts)}")
            print(all_full_divergence_stats, all_full_diversity_stats, all_subsample_divergence_stats, all_subsample_diversity_stats)
            combined_stats_list = [a + b[2:] + c[2:] + d[2:] for a, b, c, d in zip(all_full_divergence_stats, all_full_diversity_stats, all_subsample_divergence_stats, all_subsample_diversity_stats)]
            # If the path ends in tsv, save as a tsv file. Otherwise, assume a json file.
            if self.results_save_path.endswith(".tsv"):
                df = pd.DataFrame(all_divergences_and_texts, columns=["Divergence", "Text", "Response", "CSB", "Cycle", "Embedding"])
                df.drop(columns=["Embedding"], inplace=True)
                df.to_csv(self.results_save_path, index=False, sep="\t")
                print(f"Number of rows in TSV: {len(df)}")
                combined_stats_df = pd.DataFrame(combined_stats_list, columns=["Cycle", "Run", "Full Divergence Mean", "Full Divergence STD", "Full Divergence Min", "Full Divergence Max", "Full Diversity Mean", "Full Diversity STD", "Full Diversity Min", "Full Diversity Max", "Subsample Divergence Mean", "Subsample Divergence STD", "Subsample Divergence Min", "Subsample Divergence Max", "Subsample Diversity Mean", "Subsample Diversity STD", "Subsample Diversity Min", "Subsample Diversity Max"])
                combined_stats_df.to_csv(self.results_save_path.replace(".tsv", "_stats.tsv"), index=False, sep="\t")
            elif self.results_save_path.endswith(".json"):
                with open(self.results_save_path, "w") as f:
                    json.dump(all_divergences_and_texts, f)
                with open(self.results_save_path.replace(".json", "_stats.json"), "w") as f:
                    json.dump(combined_stats_list, f)
            else:
                raise ValueError("Results save path must end with .tsv or .json")
        return all_divergences_and_texts

    def search_loop(self, all_divergences_and_texts : List[Tuple[float, str, str, bool, int, List[float]]], repeat_n: int = 0) -> Tuple[List[Tuple[float, str, str, bool, int, List[float]]], List[float], List[float], List[float], List[float]]:
        if len(self.seed_demonstrations_list) > 0:
            # Compute divergences for seed_demonstrations_list
            with torch.no_grad():
                result = self.contrastive_decoder.decode(text_set = self.seed_demonstrations_list)
            divergences = result['prefix_divergences'].tolist()
            embeddings_list = get_results_embeddings_and_divergences(self.seed_demonstrations_list, self.local_embedding_model_str, self.local_embedding_model, self.local_embedding_tokenizer, self.device)

            combined_loop_divergences_and_texts = [(d, s, "", 1, 0, e) for d, s, e in zip(divergences, self.seed_demonstrations_list, embeddings_list)]
            # ^ Entries are (divergence, prompt, response, custom model score, cycle, embedding)

            print("len(divergences):", len(divergences))
            print("len(embeddings_list):", len(embeddings_list))
            print("len(self.seed_demonstrations_list):", len(self.seed_demonstrations_list))
            print("len(result['texts']):", len(result['texts']))
            print("len(combined_loop_divergences_and_texts):", len(combined_loop_divergences_and_texts))
            round_subsample_divergences_and_texts, round_subsample_divergence_stats, round_subsample_diversity_stats, round_full_divergence_stats, round_full_diversity_stats, subsample_diversity_weights, full_diversity_weights = divergence_weighted_subsample(combined_loop_divergences_and_texts, self.n_past_texts_subsampled, self.subsampling_randomness_temperature)
            if self.verbose:
                print("seed texts:")
                print_texts_with_divergences(combined_loop_divergences_and_texts, max_width = self.max_width, diversity_weights = subsample_diversity_weights)
        else:
            combined_loop_divergences_and_texts = []
        combined_loop_subsample_divergence_stats = []
        combined_loop_subsample_diversity_stats = []
        combined_loop_full_divergence_stats = []
        combined_loop_full_diversity_stats = []

        cycle_count = 0
        failed_cycle_count = 0
        while cycle_count < self.n_cycles_ask_assistant:
            n_additional_texts, round_subsample_divergences_and_texts, round_new_divergences_and_texts, round_subsample_divergence_stats, round_subsample_diversity_stats, round_full_divergence_stats, round_full_diversity_stats = self.search_step(cycle_count + 1, round_subsample_divergences_and_texts, combined_loop_divergences_and_texts, all_divergences_and_texts)
            if n_additional_texts > 0:
                round_subsample_divergence_stats = [repeat_n, cycle_count] + round_subsample_divergence_stats
                round_subsample_diversity_stats = [repeat_n, cycle_count] + round_subsample_diversity_stats
                round_full_divergence_stats = [repeat_n, cycle_count] + round_full_divergence_stats
                round_full_diversity_stats = [repeat_n, cycle_count] + round_full_diversity_stats
                cycle_count += 1
                failed_cycle_count = 0
                combined_loop_subsample_divergence_stats.append(round_subsample_divergence_stats)
                combined_loop_subsample_diversity_stats.append(round_subsample_diversity_stats)
                combined_loop_full_divergence_stats.append(round_full_divergence_stats)
                combined_loop_full_diversity_stats.append(round_full_diversity_stats)

                if self.require_unique_additional_texts:
                    # Only include additional texts that haven't been previously generated
                    round_new_divergences_and_texts = [e for e in round_new_divergences_and_texts if e[1] not in [t[1] for t in all_divergences_and_texts]]

                combined_loop_divergences_and_texts = combined_loop_divergences_and_texts + round_new_divergences_and_texts
                
            else:
                failed_cycle_count += 1
                print("Failed to get additional texts from this round. Retrying...")
                print("Failed cycle count:", failed_cycle_count)
                if failed_cycle_count >= self.max_failed_cycles:
                    raise ValueError("Maximum number of failed cycles reached. Exiting search loop.")
        try:
            # Sort the list by divergence values in descending order
            combined_loop_divergences_and_texts.sort(key=lambda x: x[0], reverse=True)

            # Print the 10 texts with the highest divergence values
            table_data = [["#", "Divergence", "Text", "Response", "CSB", "Cycle"]]
            for i in range(min(10, len(combined_loop_divergences_and_texts))):
                wrapped_text = textwrap.fill(combined_loop_divergences_and_texts[i][1].replace("\n", "\\n"), width=self.max_width)
                wrapped_response = textwrap.fill(combined_loop_divergences_and_texts[i][2].replace("\n", "\\n"), width=self.max_width)
                row = [str(i+1), str(round(combined_loop_divergences_and_texts[i][0], 3)), wrapped_text, wrapped_response, str(combined_loop_divergences_and_texts[i][3] != 0), str(combined_loop_divergences_and_texts[i][4])]
                table_data.append(row)
            table = AsciiTable(table_data, "Highest Divergence Texts")
            table.inner_row_border = True
            print(table.table)
            
            # Print the 10 texts with the lowest divergence values
            table_data_lowest = [["#", "Divergence", "Text", "Response", "CSB", "Cycle"]]
            for i in range(min(10, len(combined_loop_divergences_and_texts))):
                wrapped_text = textwrap.fill(combined_loop_divergences_and_texts[-(i+1)][1].replace("\n", "\\n"), width=self.max_width)
                wrapped_response = textwrap.fill(combined_loop_divergences_and_texts[-(i+1)][2].replace("\n", "\\n"), width=self.max_width)
                row = [str(i+1), str(round(combined_loop_divergences_and_texts[-(i+1)][0], 3)), wrapped_text, wrapped_response, str(combined_loop_divergences_and_texts[-(i+1)][3] != 0), str(combined_loop_divergences_and_texts[-(i+1)][4])]
                table_data_lowest.append(row)
            table_lowest = AsciiTable(table_data_lowest, "Lowest Divergence Texts")
            table_lowest.inner_row_border = True
            print(table_lowest.table)
        except:
            print("Error: Could not print texts with highest/lowest divergence values.")
        
        return combined_loop_divergences_and_texts, combined_loop_full_divergence_stats, combined_loop_full_diversity_stats, combined_loop_subsample_divergence_stats, combined_loop_subsample_diversity_stats

    def search_step(self, cycle_count : int, round_subsample_divergences_and_texts : List[Tuple[float, str, str, int, int, List[float]]], combined_loop_divergences_and_texts : List[Tuple[float, str, str, int, int, List[float]]], all_divergences_and_texts : List[Tuple[float, str, str, int, int, List[float]]]) -> Tuple[int, List[Tuple[float, str, str, int, int, List[float]]], List[float], List[float], List[float], List[float]]:
        # Compose messages to send the assistant
        if len(round_subsample_divergences_and_texts) > 0:
            # First, rescale round_subsample_divergences_and_texts divergence values into [0, 10]
            rescaled_round_subsample_divergences_and_texts = rescale_divergences(round_subsample_divergences_and_texts, randomize_divs = self.randomize_in_context_divs)
            if self.include_context_divs:
                # Format divergences and texts into a single string to provide the assistant
                current_divergences_and_texts_str = "\n".join([f"divergence {i}: {round(div, 3)}, input text {i}: {text}" for i, (div, text) in enumerate(rescaled_round_subsample_divergences_and_texts)])
            else:
                current_divergences_and_texts_str = "\n".join([f"input text {i}: {text}" for i, (_, text) in enumerate(rescaled_round_subsample_divergences_and_texts)])
            messages=[
            {"role": "system", "content": self.system_start_prompt + "\n" + self.system_loop_prompt},
            {"role": "user", "content": current_divergences_and_texts_str + "\n" + self.user_end_prompt}
            ]
        else:
             messages=[{"role": "system", "content": self.system_start_prompt},
            {"role": "user", "content": self.user_end_prompt}
             ]

        # Get new messages from the assistant:
        output_text = self.get_continuation(messages, self.openai_model_str, self.client, self.local_model, self.local_tokenizer, self.device, temperature=self.temperature)
        additional_texts = self.interpret_assistant_outputs(output_text)
        print("additional_texts:", additional_texts)

        round_new_divergences_and_texts = []
        if len(additional_texts) > 0:
            # Compute divergences for additional_texts
            with torch.no_grad():
                result = self.contrastive_decoder.decode(text_set = additional_texts)
            additional_divergences = result['prefix_divergences'].tolist()
            target_distribution_decodings = []
            if self.decode_only_from_comparison_model:
                with torch.no_grad():
                    # Decode from JUST the comparison model
                    for text in additional_texts:
                        encoded_prompt = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
                        n_prompt_tokens = encoded_prompt.shape[1]
                        generated = self.comparison_model.generate(
                            input_ids=encoded_prompt,
                            max_length=self.generation_length + n_prompt_tokens,
                            temperature=1.0,
                            top_p=0.9,
                            num_return_sequences=1,
                            return_dict_in_generate=True
                        )
                        returned_ids = generated.sequences[0][n_prompt_tokens:]
                        decoded_output = self.tokenizer.decode(returned_ids, skip_special_tokens=True)
                        target_distribution_decodings.append(decoded_output)
            else:
                # Use the texts generated from the contrastive decoding, selecting one text for every generations_per_prefix texts
                target_distribution_decodings = result['texts'][::self.generations_per_prefix]
            
            # Compute embeddings to allow for diversity scores
            if self.use_embeddings_diversity_score:
                embeddings_list = get_results_embeddings_and_divergences(additional_texts, self.local_embedding_model_str, self.local_embedding_model, self.local_embedding_tokenizer, self.device)
                
                print("len(embeddings_list):", len(embeddings_list))
                print("len(additional_divergences):", len(additional_divergences))
                print("len(additional_texts):", len(additional_texts))
                print("len(target_distribution_decodings):", len(target_distribution_decodings))
            else:
                embeddings_list = [None for _ in additional_texts]
            
            # Compute scores based on custom prompts given to the assistant
            custom_scores = [0 for _ in additional_divergences]
            round_new_divergences_and_texts = list(zip(additional_divergences, additional_texts, target_distribution_decodings, custom_scores, [cycle_count for _ in additional_divergences], embeddings_list))
            if self.use_custom_selection_criterion or self.require_custom_selection_for_inclusion:
                custom_scores = self.score_by_custom_selection_criterion(round_new_divergences_and_texts, self.openai_model_str, self.client, self.local_model, self.local_tokenizer, self.custom_selection_criterion, self.custom_selection_criterion_yes_response, self.custom_selection_criterion_no_response, self.custom_selection_criterion_examples, self.device)
                # if self.verbose:
                #     table_data = [["#", "Prompt", "Response", "Custom Score Boolean"]]
                #     for i, (_, prompt, response, custom_score) in enumerate(zip(additional_divergences, additional_texts, target_distribution_decodings, custom_scores)):
                #         custom_score_boolean = custom_score != 0
                #         row = [str(i+1), textwrap.fill(prompt.replace("\n", "\\n"), width=self.max_width), textwrap.fill(response.replace("\n", "\\n"), width=self.max_width), str(custom_score_boolean)]
                #         table_data.append(row)
                #     table = AsciiTable(table_data)
                #     table.inner_row_border = True
                #     print(table.table)
                round_new_divergences_and_texts = list(zip(additional_divergences, additional_texts, target_distribution_decodings, custom_scores, [cycle_count for _ in additional_divergences], embeddings_list))
                if self.require_custom_selection_for_inclusion:
                    round_new_divergences_and_texts = [e for e in round_new_divergences_and_texts if e[3] != 0]

            # Expand combined_loop_divergences_and_texts with results:
            expanded_loop_divergences_and_texts = combined_loop_divergences_and_texts + round_new_divergences_and_texts
        else:
            expanded_loop_divergences_and_texts = combined_loop_divergences_and_texts
        # Assemble a new query from the assistant's responses / additional_texts.
        # Feed the assistant responses from expanded_loop_divergences_and_texts
        # Subsample from expanded_loop_divergences_and_texts
        round_subsample_divergences_and_texts, round_subsample_divergence_stats, round_subsample_diversity_stats, round_full_divergence_stats, round_full_diversity_stats, subsample_diversity_weights, full_diversity_weights = divergence_weighted_subsample(divergences_and_texts = expanded_loop_divergences_and_texts, 
                                                                n_past_texts_subsampled = self.n_past_texts_subsampled, 
                                                                subsampling_randomness_temperature = self.subsampling_randomness_temperature, 
                                                                use_custom_selection_criterion_for_scoring = self.use_custom_selection_criterion_for_scoring, 
                                                                divergence_weight = self.divergence_weight, 
                                                                custom_selection_criterion_weight = self.custom_selection_criterion_weight, 
                                                                embedding_diversity_weight = self.embedding_diversity_weight
                                                            )
        
        n_additional_texts = len(round_new_divergences_and_texts)
        
        print("Num total texts     :", len(expanded_loop_divergences_and_texts))
        print("Num additional texts:", n_additional_texts)
        if self.verbose:
            print("round_subsample_divergences_and_texts:")
            print_texts_with_divergences(round_subsample_divergences_and_texts, max_width = self.max_width, diversity_weights = subsample_diversity_weights)
        return n_additional_texts, round_subsample_divergences_and_texts, round_new_divergences_and_texts, round_subsample_divergence_stats, round_subsample_diversity_stats, round_full_divergence_stats, round_full_diversity_stats

    def get_continuation(self, 
                         messages : List[dict], 
                         openai_model_str : Optional[str], 
                         client : Optional[OpenAI], 
                         local_model : PreTrainedModel, 
                         local_tokenizer : PreTrainedTokenizer, 
                         device : str, 
                         verbose_continuations : bool = True,
                         temperature : float = 0.0,
                         max_tokens : int = 500
                         ) -> str:
        if openai_model_str is not None:
            if client is None:
                raise ValueError("You must provide an OpenAI API key to use an OpenAI model.")
            # Generate a new text from ChatGPT
            if verbose_continuations:
                print("messages:", messages)
            chat_completion = client.chat.completions.create(
                model=openai_model_str,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            output_text = chat_completion.choices[0].message.content
        else:
            # Generate a new text from the local model
            # First, compose the different texts in messages into a single prompt
            prompt = "\n".join([message['content'] for message in messages]) + "\n\nAnswer:\n"
            if verbose_continuations:
                print("prompt:", prompt)
            prompt_tokens = local_tokenizer.encode(prompt, return_tensors="pt").to(device)
            n_prompt_tokens = prompt_tokens.shape[1]
            #json_stopping_tokens = local_tokenizer.encode("}")
            #json_stopping_criteria = StoppingCriteria(stop=json_stopping_tokens)
            with torch.no_grad():
                if temperature > 0.0:
                    output = local_model.generate(
                        input_ids=prompt_tokens,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        do_sample=True,
                        top_p=0.95,
                        top_k=0,
                        num_return_sequences=1,
                        return_dict_in_generate=True
                    )
                else:
                    output = local_model.generate(
                        input_ids=prompt_tokens,
                        max_new_tokens=max_tokens,
                        do_sample=False,
                        num_return_sequences=1,
                        return_dict_in_generate=True
                    )
            output_tokens = output.sequences[0][n_prompt_tokens:]
            output_text = local_tokenizer.decode(output_tokens, skip_special_tokens=True)

        return output_text
    
    # This function takes in a list of prompts and responses, and returns a list of scores
    # where each score either 0 or 1, indicating whether each prompt satisfied the custom 
    # selection criterion.
    def score_by_custom_selection_criterion(self, 
                                            prompts_and_responses : List[Tuple[int, str, str, int, List[float]]], 
                                            openai_model_str : Optional[str], 
                                            client : Optional[OpenAI], 
                                            local_model : PreTrainedModel, 
                                            local_tokenizer : PreTrainedTokenizer, 
                                            custom_selection_criterion : Optional[str], 
                                            yes_response_strs : List[str], 
                                            no_response_strs : List[str], 
                                            custom_selection_criterion_examples : Optional[List[str]], 
                                            device : str
                                            ) -> List[int]:
        scores = []
        for _, prompt, response, _, _, _ in prompts_and_responses:
            # Combine the prompt and response into a single string
            combined_text = f"Prompt: {prompt}\nResponse: {response}"
            # Create the messages
            messages = [{"role": "system", "content": custom_selection_criterion}]
            if custom_selection_criterion_examples is not None:
                for example in custom_selection_criterion_examples:
                    messages.append({"role": "user", "content": example})
            messages.append({"role": "user", "content": combined_text})
            # Query the assistant model
            assistant_response = self.get_continuation(
                messages=messages,
                openai_model_str=openai_model_str,
                client=client,
                local_model=local_model,
                local_tokenizer=local_tokenizer,
                device=device,
                temperature=0.0,
                max_tokens=20
            )
            # Determine if the assistant's response matches the expected "yes" or "no" responses
            assistant_response_bool = False
            match_found = False
            response_len = len(assistant_response)
            processed_assistant_response = assistant_response[:min(response_len, 5)].lower()
            print(processed_assistant_response)
            for yes_response in yes_response_strs:
                if yes_response in processed_assistant_response:
                    assistant_response_bool = True
                    match_found = True
                    break
            for no_response in no_response_strs:
                if no_response in processed_assistant_response:
                    assistant_response_bool = False
                    if match_found:
                        print("Conflicting matches found for processed_assistant_response:", processed_assistant_response)
                        print("Yes response:", yes_response)
                        print("No response:", no_response)
                    match_found = True
                    break
            if match_found:
                scores.append(assistant_response_bool)
            else:
                # If the response is neither "yes" nor "no", log an error and append a default score
                print(f"Error: Unexpected response from assistant model: {assistant_response}")
                scores.append(0)  # Default score for unexpected responses
        return scores

    # This function takes in a string of text and interprets it as a json object, returning a list of texts.
    # If the json parsing fails, it tries to interpret the text as a list of texts separated by newlines, and returns a list of texts.
    #
    # output_text: The string of text to interpret.
    # as_json: Whether to interpret the text as a json object or a newline-separated list of texts.
    # fallback_to_newlines: Whether to interpret the text as a newline-separated list of texts if the json parsing fails.
    def interpret_assistant_outputs(self,
                                    output_text : str, 
                                    as_json : bool = True, 
                                    fallback_to_newlines : bool = False) -> List[str]:
        json_parse_fail = False
        prompts = []
        if as_json:
            # Interpret output_text as a json object with a dictionary and parse it:
            # First, remove all characters before the first '{' character
            filtered_output_text = output_text[output_text.find('{'):]
            # Then, find the corresponding closing '}' character
            end_index = find_matching_brace(filtered_output_text)
            if end_index != -1:
                filtered_output_text = filtered_output_text[:end_index+1]
            else:
                print("Error: Could not find matching '}' character.")
                json_parse_fail = True
            try:
                output_text_json = json.loads(filtered_output_text)
            except:
                print("Error: Could not parse output_text as a json object.")
                print(output_text)
                print("Attempting to use assistant to extract json...")
                extraction_prompt = f"Extract the json object from the following text, and say nothing but the json object:\n{output_text}"
                messages = [{"role": "system", "content": extraction_prompt}]
                assistant_extract_json = self.get_continuation(messages, self.openai_model_str, self.client, self.local_model, self.local_tokenizer, self.device, temperature=0.0)
                output_text = assistant_extract_json
                try:
                    filtered_assistant_extract_json = assistant_extract_json[assistant_extract_json.find('{'):]
                    filtered_assistant_extract_json = filtered_assistant_extract_json[:filtered_assistant_extract_json.rfind('}')+1]
                    output_text_json = json.loads(filtered_assistant_extract_json)
                except:
                    print("Error: Could not parse assistant output as a json object.")
                    print(assistant_extract_json)
                    json_parse_fail = True
            if not json_parse_fail:
                # Extract all entries from the dictionary
                entries = output_text_json.items()
                #print("output_text_json", output_text_json)
                # Filter out entries that do not start with "prompt"
                prompt_entries = [entry for entry in entries if ("prompt" in entry[0].lower() or "response" in entry[0].lower())]
                # Sort the prompt entries by the number at the end of the key (e.g. "prompt 1", "prompt 2", ...)
                prompt_entries = sorted(prompt_entries, key=lambda x: int(re.search(r'\d+', x[0]).group()))
                # Extract the values (i.e. the texts) from the prompt entries
                prompts = [entry[1] for entry in prompt_entries]
        if not as_json or (json_parse_fail and fallback_to_newlines):
            # Interpret output_text as a list of texts separated by newlines
            if json_parse_fail:
                # Try to strip out any json-related characters and split by newlines
                output_text = re.sub(r'[^a-zA-Z0-9\s]', '', output_text)
                output_text = output_text.replace("\n\n", "\n")
            prompts = output_text.split("\n")
        return prompts

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        return get_results_embeddings_and_divergences(
            texts,
            self.local_embedding_model_str,
            self.local_embedding_model,
            self.local_tokenizer,
            self.device
        )
        

# This function takes in a list of divergences / texts / response / custom_score tuples, and rescales the divergences into [0, max_div].
# It returns a list of tuples of divergences / texts / response / custom_score tuples, where the texts are the same as the input texts.
# If randomize_divs is True, it will randomize the divergences between 0 and max_div.
#
# divergences_and_texts: A list of tuples of divergences / texts / response / custom_score tuples.
def rescale_divergences(divergences_and_texts : List[Tuple[float, str, str, int, int, List[float]]], max_div : float = 10, randomize_divs : bool = False) -> List[Tuple[float, str]]:
    if randomize_divs:
        texts = [e[1] for e in divergences_and_texts]
        random_round_divergences_and_texts = [(max_div * np.random.random(), text) for text in texts]
        min_divergence = min(random_round_divergences_and_texts, key=lambda x: x[0])[0]
        max_divergence = max(random_round_divergences_and_texts, key=lambda x: x[0])[0]
        rescaled_round_divergences_and_texts = [(max_div * (div - min_divergence) / max((max_divergence - min_divergence), 0.0001), text) for div, text in random_round_divergences_and_texts]
        # Now re-sort the rescaled divergences and texts by divergence
        rescaled_round_divergences_and_texts = sorted(rescaled_round_divergences_and_texts, key=lambda x: x[0], reverse=True)
    else:
        max_divergence = max(divergences_and_texts, key=lambda x: x[0])[0]
        min_divergence = min(divergences_and_texts, key=lambda x: x[0])[0]
        rescaled_round_divergences_and_texts = [(max_div * (div - min_divergence) / max((max_divergence - min_divergence), 0.0001), text) for div, text, _, _, _, _ in divergences_and_texts]
    return rescaled_round_divergences_and_texts



    

# This function takes in a divergences_and_texts list, and returns a random subsample of that list,
# weighted by the divergence values, with subsampling_randomness_temperature scaling the degree of randomness.
# Optionally takes in a boolean to use scores from a custom selection criterion to bias selection process.
# It also optionally can use the embeddings in the divergences_and_texts list to select diverse texts,
# operationalized as a preference for texts whose embeddings are far from other texts' embeddings using 
# the L 1/2 norm.
def divergence_weighted_subsample(divergences_and_texts : List[Tuple[float, str, str, int, int, List[float]]], 
                                  n_past_texts_subsampled : int = 5, 
                                  subsampling_randomness_temperature : float = 0.5, 
                                  select_high_divergence : bool = True, 
                                  use_custom_selection_criterion_for_scoring : bool = False,
                                  use_embeddings_for_diversity : bool = True,
                                  divergence_weight : float = 1.0,
                                  embedding_diversity_weight : float = 1.0,
                                  custom_selection_criterion_weight : float = 1.0,
                                  sort_output : bool = True
                                  ) -> List[Tuple[float, str, str, int, int, List[float]]]:
    if len(divergences_and_texts) == 0:
        return []
    
    # First, get the divergence's contribution to the weights, rescaled to be between 0 and 1
    weights = [t[0] * divergence_weight for t in rescale_divergences(divergences_and_texts, max_div = 1)]

    # Add the custom selection criterion to the weights
    if use_custom_selection_criterion_for_scoring:
        weights = [w + custom_selection_criterion_weight * entry[3] for w, entry in zip(weights, divergences_and_texts)]
    
    # Calculate the all-pairs L2 distances between the embeddings
    embeddings = np.array([entry[5] for entry in divergences_and_texts])
    embedding_distances_matrix = np.linalg.norm(embeddings[:, np.newaxis] - embeddings[np.newaxis, :], ord=2, axis=-1)
    # Compute the harmonic mean of each embedding's distance to all other embeddings
    # (We use the harmonic mean to prevent large distances from having a disproportionately large weight)
    # Need to deal with diagonal elements of the matrix, which will otherwise throw off the estimates.
    # The harmonic mean is dominated by the *smallest* values, so we set them to inf.
    embedding_distances_matrix[np.diag_indices_from(embedding_distances_matrix)] = np.inf
    embedding_diversity_weights = np.reciprocal(np.mean(np.reciprocal(embedding_distances_matrix + 1e-8), axis=1))
    # Rescale the diversity weights to be between 0 and 1
    embedding_diversity_weights_scaled = (embedding_diversity_weights - np.min(embedding_diversity_weights)) / (np.max(embedding_diversity_weights) - np.min(embedding_diversity_weights))
    
    if use_embeddings_for_diversity:
        # Add the diversity contribution to the weights
        weights = [w + embedding_diversity_weight * ed for w, ed in zip(weights, embedding_diversity_weights_scaled)]

    # Scale the weights to ensure they fall into [0, 1]
    min_weight = min(weights)
    max_weight = max(weights)
    weights = [(w - min_weight) / max(max_weight - min_weight, 1e-7) for w in weights]

    if subsampling_randomness_temperature > 0:
        if select_high_divergence:
            weights = np.exp(np.array(weights) / subsampling_randomness_temperature)
        else:
            weights = np.exp(-np.array(weights) / subsampling_randomness_temperature)
        weights /= np.sum(weights)

        n_samples = min(n_past_texts_subsampled, len(divergences_and_texts))
        # Randomly choose indices based on weights without repetition
        try:
            chosen_indices = np.random.choice(len(divergences_and_texts), size=n_samples, p=weights, replace=False)
        except:
            print("weights:", weights)
            print("n_samples:", n_samples)
            raise ValueError("Could not select indices based on weights.")
    else:
        # If no randomness, just select the top n_past_texts_subsampled texts by weights
        chosen_indices = np.argsort(weights)[-n_samples:]

    # Select the subsampled pairs
    subsample = [divergences_and_texts[i] for i in chosen_indices]
    if sort_output:
        # Sort the subsample by divergence
        subsample = sorted(subsample, key=lambda x: x[0], reverse=True)
    
    subsample_diversity_weights = embedding_diversity_weights[chosen_indices]

    # Compute divergence and diversity stats
    subsample_divergence_stats = [
        np.mean([entry[0] for entry in subsample]),
        np.std([entry[0] for entry in subsample]),
        np.max([entry[0] for entry in subsample]),
        np.min([entry[0] for entry in subsample])
    ]
    subsample_diversity_stats = [
        np.mean(subsample_diversity_weights),
        np.std(subsample_diversity_weights),
        np.max(subsample_diversity_weights),
        np.min(subsample_diversity_weights)
    ]

    full_divergence_stats = [
        np.mean([entry[0] for entry in divergences_and_texts]),
        np.std([entry[0] for entry in divergences_and_texts]),
        np.max([entry[0] for entry in divergences_and_texts]),
        np.min([entry[0] for entry in divergences_and_texts])
    ]
    full_diversity_stats = [
        np.mean(embedding_diversity_weights),
        np.std(embedding_diversity_weights),
        np.max(embedding_diversity_weights),
        np.min(embedding_diversity_weights)
    ]

    # Return the subsample
    return subsample, subsample_divergence_stats, subsample_diversity_stats, full_divergence_stats, full_diversity_stats, subsample_diversity_weights, embedding_diversity_weights

def print_texts_with_divergences(divergences_and_texts : List[Tuple[float, str, str, int, int, List[float]]], max_width : int = 50, diversity_weights : List[float] = None) -> None:
    if diversity_weights is None:
        table_data = [["Div", "Text", "Response", "CSB", "Cycle"]]
        for div, text, response, custom_score, cycle, _ in divergences_and_texts:
            wrapped_text = "\n".join(textwrap.wrap(text, width=max_width))
            wrapped_response = "\n".join(textwrap.wrap(response, width=max_width))
            table_data.append([round(div, 3), wrapped_text, wrapped_response, str(custom_score != 0), str(cycle)])
    else:
        table_data = [["Div", "Text", "Response", "CSB", "Cycle", "Diversity"]]
        for (div, text, response, custom_score, cycle, _), diversity_weights in zip(divergences_and_texts, diversity_weights):
            wrapped_text = "\n".join(textwrap.wrap(text, width=max_width))
            wrapped_response = "\n".join(textwrap.wrap(response, width=max_width))
            table_data.append([round(div, 3), wrapped_text, wrapped_response, str(custom_score != 0), str(cycle), str(round(diversity_weights, 3))])
    table = AsciiTable(table_data)
    table.inner_row_border = True
    print(table.table)

def find_matching_brace(text, start_index = 0):
    stack = []
    for i, char in enumerate(text[start_index:], start=start_index):
        if char == '{':
            stack.append(char)
        elif char == '}':
            stack.pop()
            if not stack:
                return i
    return -1