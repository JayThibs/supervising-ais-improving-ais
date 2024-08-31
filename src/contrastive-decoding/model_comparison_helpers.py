import torch
import shutil
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils.logging import set_verbosity_error
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from pandas import read_csv
from transformers import PreTrainedTokenizer
from typing import Optional, List, Tuple, Type, Dict
from tqdm import tqdm
from transformers import BitsAndBytesConfig

from transformers import (
    GPT2LMHeadModel,
    OPTForCausalLM,
    LlamaForCausalLM,
    MistralForCausalLM,
    GPTNeoXForCausalLM,
    GPTJForCausalLM,
    GPT2LMHeadModel
)

from collections import OrderedDict
import json
from pathlib import Path
from collections import defaultdict
import pickle

class LRUCache(OrderedDict):
    def __init__(self, max_size : int = 2):
        self.max_size = max_size
        super().__init__()

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if len(self) > self.max_size:
            oldest = next(iter(self))
            del self[oldest]

# Adapted from: https://github.com/xiamengzhou/training_trajectory_analysis/blob/main/utils.py
def build_contrastive_lm(superclass : Type[PreTrainedModel]) -> Type[PreTrainedModel]:
    class CausalLMSubtract(superclass):
        def __init__(self, config, **kwargs):
            print("Initializing ContrastiveLM")
            # Extract custom parameters
            self.starting_model_weight = kwargs.pop('starting_model_weight', 1)
            self.comparison_model_weight = kwargs.pop('comparison_model_weight', -1)
            self.limit_to_starting_model_top_p = kwargs.pop('limit_to_starting_model_top_p', None)
            self.similarity_gating_intensity = kwargs.pop('similarity_gating_intensity', None)
            self.comparison_model_prefix_ids = kwargs.pop('comparison_model_prefix_ids', None)
            self.starting_model_prefix_ids = kwargs.pop('starting_model_prefix_ids', None)
            self.cache_attn = kwargs.pop('cache_attn', False)

            # Initialize the base class
            super().__init__(config)

            self.comparison_lm = None
            self.comparison_model_prefix_ids_added = False
            self.starting_model_prefix_ids_added = False

            self.past_starting_attn_storage = LRUCache(max_size=2)
            self.past_comparison_attn_storage = LRUCache(max_size=2)

            if self.comparison_model_prefix_ids is not None:
                self.n_comparison_model_prefix_ids = self.comparison_model_prefix_ids.size()[-1]
            else:
                self.n_comparison_model_prefix_ids = 0
            if self.starting_model_prefix_ids is not None:
                self.n_starting_model_prefix_ids = self.starting_model_prefix_ids.size()[-1]
            else:
                self.n_starting_model_prefix_ids = 0
                
        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
                print(f"Calling from_pretrained for {cls.__name__}")
                model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
                
                # Set custom attributes
                model.starting_model_weight = kwargs.get('starting_model_weight', 1)
                model.comparison_model_weight = kwargs.get('comparison_model_weight', -1)
                model.limit_to_starting_model_top_p = kwargs.get('limit_to_starting_model_top_p', None)
                model.similarity_gating_intensity = kwargs.get('similarity_gating_intensity', None)
                model.comparison_model_prefix_ids = kwargs.get('comparison_model_prefix_ids', None)
                model.starting_model_prefix_ids = kwargs.get('starting_model_prefix_ids', None)
                model.cache_attn = kwargs.get('cache_attn', False)

                return model
            
        def forward(self, **kwargs):
            starting_model_input = kwargs.copy()
            comparison_model_input = kwargs.copy()

            # Handle past_key_values
            if 'past_key_values' in kwargs and kwargs['past_key_values'] is not None:
                if isinstance(kwargs['past_key_values'], tuple) and len(kwargs['past_key_values']) == 2:
                    starting_model_input['past_key_values'] = kwargs['past_key_values'][0]
                    comparison_model_input['past_key_values'] = kwargs['past_key_values'][1]
                else:
                    starting_model_input.pop('past_key_values', None)
                    comparison_model_input.pop('past_key_values', None)

            # Ensure inputs are on the correct device
            starting_model_input = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                    for k, v in starting_model_input.items()}
            
            starting_model_output = super().forward(**starting_model_input)

            if self.comparison_lm is not None:
                comparison_model_input = {k: v.to(self.comparison_lm.device) if isinstance(v, torch.Tensor) else v 
                                        for k, v in comparison_model_input.items()}
                comparison_model_output = self.comparison_lm(**comparison_model_input)
                comparison_model_logits = comparison_model_output.logits.to(self.device)
            else:
                comparison_model_logits = torch.zeros_like(starting_model_output.logits)

            # Store the comparison_model_logits for use in calculate_current_divergence
            self._last_comparison_model_logits = comparison_model_logits

            # Return CausalLMOutputWithPast for compatibility with generate method
            return CausalLMOutputWithPast(
                loss=starting_model_output.loss,
                logits=starting_model_output.logits,
                past_key_values=starting_model_output.past_key_values,
                hidden_states=starting_model_output.hidden_states,
                attentions=starting_model_output.attentions,
            )

        def calculate_current_divergence(self, 
                                         text_ids: torch.Tensor, 
                                         batch_size: int = 16, 
                                         end_tokens_to_only_consider: int = 0,
                                         return_perplexities: bool = False,
                                         return_all_token_divergences: bool = False,
                                         return_all_vocab_divergences: bool = False,
                                         logits_comparison_top_p: Optional[float] = None,
                                         use_avg_KL_as_divergences: bool = True,
                                         KL_distribution_choice: str = "auto",
                                         progress_bar: bool = False
                                        ) -> dict:
            print(f"Calculating divergences for {self.__class__.__name__}")
            n_texts = len(text_ids)
            n_batches = n_texts // batch_size + (n_texts % batch_size != 0)
            divergences = []
            all_token_divergences = []
            all_vocab_divergences = []
            starting_model_perplexities = []
            comparison_model_perplexities = []
            
            for i in tqdm(range(n_batches), disable=not progress_bar, desc="Calculating divergences"):
                batch_start = i * batch_size
                batch_end = min((i + 1) * batch_size, n_texts)
                current_ids = text_ids[batch_start : batch_end]
                
                # Forward pass without labels to avoid the size mismatch error
                output = self(input_ids=current_ids)
                
                starting_model_logits = output.logits
                comparison_model_logits = self._last_comparison_model_logits
    
                vocab_size = starting_model_logits.size()[-1]
                n_tokens = starting_model_logits.size()[-2]

                if comparison_model_logits.size()[-1] > starting_model_logits.size()[-1]:
                    comparison_model_logits = comparison_model_logits[:, :, : vocab_size]

                div_starting_model_logits = starting_model_logits.clone()
                div_comparison_model_logits = comparison_model_logits.clone()
                
                if end_tokens_to_only_consider > 0:
                    div_starting_model_logits = starting_model_logits[:, -end_tokens_to_only_consider:, :]
                    div_comparison_model_logits = comparison_model_logits[:, -end_tokens_to_only_consider:, :]

                if use_avg_KL_as_divergences:
                    if KL_distribution_choice == "auto":
                        if self.starting_model_weight < 0 and self.comparison_model_weight > 0:
                            token_probabilities = torch.softmax(div_comparison_model_logits, dim=-1)
                        elif self.starting_model_weight > 0 and self.comparison_model_weight < 0:
                            token_probabilities = torch.softmax(div_starting_model_logits, dim=-1)
                        else:
                            token_probabilities = torch.softmax(self.starting_model_weight * div_starting_model_logits + self.comparison_model_weight * div_comparison_model_logits, dim=-1)
                    elif "comparison" in str.lower(KL_distribution_choice):
                        token_probabilities = torch.softmax(div_comparison_model_logits, dim=-1)
                    elif "starting" in str.lower(KL_distribution_choice):
                        token_probabilities = torch.softmax(div_starting_model_logits, dim=-1)
                    else:
                        raise ValueError("KL_distribution_choice not recognized.")
                    prob_weighted_token_divergences = torch.sum((self.starting_model_weight * div_starting_model_logits + self.comparison_model_weight * div_comparison_model_logits) * token_probabilities, dim=-1)
                    
                    token_correction_factors = self.starting_model_weight * torch.logsumexp(div_starting_model_logits, dim=-1) + self.comparison_model_weight * torch.logsumexp(div_comparison_model_logits, dim=-1)

                    token_divergences = prob_weighted_token_divergences - token_correction_factors
                    batch_divergences = torch.mean(token_divergences, dim=-1).tolist()
                    token_divergences = token_divergences.tolist()

                    if return_all_vocab_divergences:
                        prob_weighted_vocab_divergences = (self.starting_model_weight * div_starting_model_logits + self.comparison_model_weight * div_comparison_model_logits) * token_probabilities
                        vocab_divergences = prob_weighted_vocab_divergences - token_correction_factors.unsqueeze(-1)
                        vocab_divergences = vocab_divergences.tolist()
                    
                else:
                    loss_fct = torch.nn.L1Loss()
                    # Computes one divergence value per id sequence in the batch (output is of size batch_size):
                    batch_divergences = [loss_fct(div_comparison_model_logits[i,:,:], div_starting_model_logits[i,:,:]).item() for i in range(len(div_comparison_model_logits))]
                    if return_all_token_divergences:
                        # Computes one divergence value for every token in every sequence in the batch (output is of size batch_size x sequence_length):
                        token_divergences = [[loss_fct(div_comparison_model_logits[i,j,:], div_starting_model_logits[i,j,:]).item() for j in range(div_comparison_model_logits.size()[1])] for i in range(len(div_comparison_model_logits))]
                
                divergences.extend(batch_divergences)
                
                if return_all_token_divergences:
                    all_token_divergences.extend(token_divergences)
                if return_all_vocab_divergences:
                    all_vocab_divergences.extend(vocab_divergences)

                if return_perplexities:
                    starting_model_batch_losses = torch.tensor([LM_loss(ids, logits, vocab_size).item() for logits,ids in zip(starting_model_logits, current_ids)])
                    comparison_model_batch_losses = torch.tensor([LM_loss(ids, logits, vocab_size).item() for logits,ids in zip(comparison_model_logits, current_ids)])
                    if end_tokens_to_only_consider > 0:
                        starting_model_batch_losses = starting_model_batch_losses * (n_tokens / (end_tokens_to_only_consider))
                        comparison_model_batch_losses = comparison_model_batch_losses * (n_tokens / (end_tokens_to_only_consider))

                    starting_model_batch_perplexities = torch.exp(starting_model_batch_losses).tolist()
                    comparison_model_batch_perplexities = torch.exp(comparison_model_batch_losses).tolist()

                    starting_model_perplexities.extend(starting_model_batch_perplexities)
                    comparison_model_perplexities.extend(comparison_model_batch_perplexities)

            result = {'divergences': divergences[:len(divergences)]}
            if return_perplexities:
                result['starting_model_perplexities'] = starting_model_perplexities[:len(divergences)]
                result['comparison_model_perplexities'] = comparison_model_perplexities[:len(divergences)]
            if return_all_token_divergences:
                result['all_token_divergences'] = all_token_divergences[:len(divergences)]
            if return_all_vocab_divergences:
                result['all_vocab_divergences'] = all_vocab_divergences[:len(divergences)]
            return result
    return CausalLMSubtract

# Adapted from: https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/models/mistral/modeling_mistral.py#L931
def LM_loss(labels : torch.Tensor, 
            logits : torch.Tensor, 
            vocab_size : int
            ) -> torch.Tensor:
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    loss = loss_fct(shift_logits, shift_labels)
    return loss


def get_cls(model_name: str) -> Type[PreTrainedModel]:
    model_name = str.lower(model_name)
    if "gpt2" in model_name:
        return GPT2LMHeadModel
    if "gptj" in model_name:
        return GPTJForCausalLM
    if "opt" in model_name:
        return OPTForCausalLM
    if "llama" in model_name or "solar" in model_name:
        return LlamaForCausalLM
    if "mistral" in model_name:
        return MistralForCausalLM
    if "gptneox" in model_name:
        return GPTNeoXForCausalLM
    if "smollm" in model_name:
        return AutoModelForCausalLM
    # If none of the above conditions are met, use AutoModelForCausalLM as a fallback
    return AutoModelForCausalLM

def instantiate_models(
        model_name : str = "gpt2-xl",
        starting_model_path : str = "gpt2-xl",
        comparison_model_path : str = "gpt2-xl",
        starting_model_weight : float = -1,
        comparison_model_weight : float = 1,
        tokenizer_family : str = "gpt2",
        device : Optional[str] = "auto",
        starting_model_device : Optional[str] = None,
        comparison_model_device : Optional[str] = None,
        limit_to_starting_model_top_p : Optional[float] = None,
        similarity_gating_intensity : Optional[float] = None,
        comparison_model_prefix_ids : Optional[List[int]] = None,
        starting_model_prefix_ids : Optional[List[int]] = None,
        no_quantize_starting_model : bool = False,
        bnb_config : Optional[BitsAndBytesConfig] = None,
        cache_attn : bool = False,
        comparison_model_interpolation_weight : Optional[float] = None,
        ) -> Tuple[PreTrainedModel, PreTrainedModel, PreTrainedTokenizer]:

    # Determine the device to use
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Using device: {device}")

    # Disable quantization for MPS
    if device == "mps":
        no_quantize_starting_model = True
        bnb_config = None

    # Set up device maps
    if device == "mps" or device == "cpu":
        starting_model_device_map = device
        comparison_model_device_map = device
    else:
        if starting_model_device is not None and comparison_model_device is not None:
            starting_model_device_map = {"": starting_model_device}
            comparison_model_device_map = {"": comparison_model_device}
        else:
            starting_model_device_map = {"": device}
            comparison_model_device_map = {"": device}
    
    print("Starting model device_map:", starting_model_device_map)
    print("Comparison model device_map:", comparison_model_device_map)

    # Load the comparison model if needed
    if comparison_model_weight != 0 or (comparison_model_interpolation_weight is not None and comparison_model_interpolation_weight != 0):
        comparison_model = AutoModelForCausalLM.from_pretrained(comparison_model_path, 
                                                                device_map=comparison_model_device_map,
                                                                quantization_config=bnb_config if not no_quantize_starting_model else None)
        comparison_model = comparison_model.eval()
    else:
        comparison_model = None

    model_class = get_cls(model_name)
    print(f"Model class: {model_class.__name__}")
    set_verbosity_error()

    # Load the starting model
    ContrastiveLM = build_contrastive_lm(model_class)
    print(f"ContrastiveLM class: {ContrastiveLM.__name__}")
    model = ContrastiveLM.from_pretrained(
        starting_model_path,
        quantization_config=bnb_config if not no_quantize_starting_model else None,
        device_map=starting_model_device_map,
        starting_model_weight=starting_model_weight,
        comparison_model_weight=comparison_model_weight,
        limit_to_starting_model_top_p=limit_to_starting_model_top_p,
        similarity_gating_intensity=similarity_gating_intensity,
        comparison_model_prefix_ids=comparison_model_prefix_ids,
        starting_model_prefix_ids=starting_model_prefix_ids,
        cache_attn=cache_attn
    )
    print(f"Model: {model.__class__.__name__}")
    print(f"Model methods: {[method for method in dir(model) if not method.startswith('__')]}")
    model = model.eval()
    model.comparison_lm = comparison_model

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_family)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    # Interpolate model weights if needed
    if comparison_model_interpolation_weight is not None and comparison_model_interpolation_weight != 0:
        starting_model_params = [p for n, p in model.named_parameters() if not "comparison_lm" in n]
        comparison_model_params = [p for n, p in model.named_parameters() if "comparison_lm" in n]
        for sparam, cparam in zip(starting_model_params, comparison_model_params):
            before_dtype = sparam.data.dtype
            integer_dtype = before_dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.quint8, torch.qint8, torch.qint32, torch.quint2x4, torch.quint4x2]
            full_precision_sparam = sparam.data.to(torch.float32)  # Changed from float64 to float32
            full_precision_cparam = cparam.data.to(torch.float32)  # Changed from float64 to float32
            sparam.data = comparison_model_interpolation_weight * full_precision_sparam + (1 - comparison_model_interpolation_weight) * full_precision_cparam
            if not integer_dtype:
                sparam.data = sparam.data.to(before_dtype)
            else:
                sparam.data = sparam.data.round().to(before_dtype)
        if comparison_model_weight == 0:
            del model.comparison_lm
            model.comparison_lm = None
            comparison_model = None

    return model, comparison_model, tokenizer

def load_jsonl_data(data_dir: str = "data/evals/anthropic-model-written-evals", 
                    selected_keys: Optional[Dict[str, List[str]]] = None,
                    interactive: bool = False,
                    save_selection: bool = False) -> List[str]:
    """
    Load data from JSONL files in the specified directory and its subdirectories.
    
    Args:
    - data_dir (str): Path to the directory containing JSONL files.
    - selected_keys (Optional[Dict[str, List[str]]]): Dictionary of file patterns and their selected keys.
    - interactive (bool): If True, prompt the user to select keys for each file pattern.
    - save_selection (bool): If True, save the key selection for future use.
    
    Returns:
    - List[str]: A list of prompts extracted from the JSONL files.
    """
    data_path = Path(data_dir)
    prompts = []
    file_patterns = defaultdict(set)
    
    # First pass: identify all unique key sets
    for jsonl_file in data_path.rglob("*.jsonl"):
        with jsonl_file.open() as f:
            keys = set()
            for line in f:
                try:
                    data = json.loads(line)
                    keys.update(data.keys())
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {jsonl_file}")
                    continue
            file_patterns[tuple(sorted(keys))].add(str(jsonl_file.relative_to(data_path)))

    if interactive or selected_keys is None:
        selected_keys = {}
        for keys, files in file_patterns.items():
            print(f"\nFound files with keys: {', '.join(keys)}")
            print(f"Example files: {', '.join(list(files)[:5])}")
            selected = input("Enter the keys you want to include (comma-separated), or press Enter to skip: ").strip()
            if selected:
                selected_keys[','.join(keys)] = [k.strip() for k in selected.split(',')]

        if save_selection:
            with open('key_selection.pkl', 'wb') as f:
                pickle.dump(selected_keys, f)
            print("Key selection saved for future use.")
    
    # Second pass: extract data based on selected keys
    for jsonl_file in data_path.rglob("*.jsonl"):
        with jsonl_file.open() as f:
            for line in f:
                try:
                    data = json.loads(line)
                    keys = ','.join(sorted(data.keys()))
                    if keys in selected_keys:
                        for key in selected_keys[keys]:
                            if key in data:
                                prompts.append(data[key])
                                break
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {jsonl_file}")
                    continue
    
    return prompts

# Modify the get_input_ids function to include the new parameters
def get_input_ids(
        tokenizer: PreTrainedTokenizer,
        single_prefix: Optional[str] = None,
        text_set: Optional[List[str]] = None,
        prefixes_path: Optional[str] = None,
        set_prefix_len: Optional[int] = None,
        n_prefixes: Optional[int] = None,
        device: str = "cuda:0",
        use_jsonl_data: bool = False,
        jsonl_data_dir: str = "data/evals/anthropic-model-written-evals",
        selected_keys: Optional[Dict[str, List[str]]] = None,
        interactive: bool = False,
        save_selection: bool = False
        ) -> torch.Tensor:
    """
    Generates input IDs from text inputs using a specified tokenizer.

    This function now supports loading data from JSONL files with flexible key selection.

    New Parameters:
    - selected_keys (Optional[Dict[str, List[str]]]): Dictionary of file patterns and their selected keys.
    - interactive (bool): If True, prompt the user to select keys for each file pattern.
    - save_selection (bool): If True, save the key selection for future use.
    """
    if use_jsonl_data:
        prompt = load_jsonl_data(jsonl_data_dir, selected_keys, interactive, save_selection)
    elif not single_prefix is None:
        prompt = [single_prefix]
        if not n_prefixes is None:
            prompt = [single_prefix] * n_prefixes
    elif not prefixes_path is None:
        if ".txt" in prefixes_path:
            prompt = open(prefixes_path, "r").readlines()
            prompt = [p.replace("\n", "") for p in prompt]
        elif ".csv" in prefixes_path:
            prompt = read_csv(prefixes_path).values[:,1].tolist()
    elif not text_set is None:
        prompt = text_set
    else:
        raise ValueError("No input method specified.")
    
    input_ids = tokenizer.batch_encode_plus(prompt, padding=True, truncation=True, return_tensors="pt", max_length=set_prefix_len)['input_ids'].to(device)
    if not n_prefixes is None:
        input_ids = input_ids[:n_prefixes]
    return input_ids

# Color tokens red in proportion to token_scores
def string_with_token_colors(text : str, 
                             token_scores : list, 
                             tokenizer : PreTrainedTokenizer,
                             min_score : float = None,
                             max_score : float = None
                             ) -> str:
    # First, tokenize the text.
    tokens = tokenizer.tokenize(text)[-len(token_scores):]
    if len(token_scores) != len(tokens):
        print("Len mismatch", tokenizer.tokenize(text))
        print(f"tokens len: {len(tokenizer.tokenize(text))}, token_scores len: {len(token_scores)}")
        return text
    # Check that we have as many tokens as token_scores
    assert len(tokens) == len(token_scores)
    
    # Now, generate normalized token scores in [0,1]
    if min_score is None or max_score is None:
        token_scores = [t - min(token_scores) for t in token_scores]
        token_scores = [t / max(max(token_scores), 1e-7) for t in token_scores]
    else:
        if max_score == 0:
            raise ValueError("max_score cannot be 0")
        token_scores = [t - min_score for t in token_scores]
        token_scores = [t / max_score for t in token_scores]

    # Now, create a string with the tokens colored red in proportion to token_scores using terminal colors
    colored_str = ""
    for token, score in zip(tokens, token_scores):
        colored_str += f"\033[48;2;{int(255*score)};0;0m{token}\033[0m"
    return colored_str