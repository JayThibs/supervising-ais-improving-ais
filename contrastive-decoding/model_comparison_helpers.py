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
from typing import Optional, List, Tuple, Type

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
        def __init__(self,
                     config,
                     starting_model_weight : float = 1,
                     comparison_model_weight : float = -1,
                     limit_to_starting_model_top_p : Optional[float] = None,
                     similarity_gating_intensity : Optional[float] = None,
                     comparison_model_prefix_ids : Optional[List[int]] = None,
                     starting_model_prefix_ids : Optional[List[int]] = None,
                     bnb_config : Optional[dict] = None,
                     cache_attn : bool = False):
            super().__init__(config)
            self.comparison_lm = None
            self.starting_model_weight = starting_model_weight
            self.comparison_model_weight = comparison_model_weight

            self.similarity_gating_intensity = similarity_gating_intensity
            self.limit_to_starting_model_top_p = limit_to_starting_model_top_p

            self.comparison_model_prefix_ids = comparison_model_prefix_ids
            self.starting_model_prefix_ids = starting_model_prefix_ids

            self.comparison_model_prefix_ids_added = False
            self.starting_model_prefix_ids_added = False

            self.past_starting_attn_storage = LRUCache(max_size=2)
            self.past_comparison_attn_storage = LRUCache(max_size=2)
            self.cache_attn = cache_attn

            if not comparison_model_prefix_ids is None:
                self.n_comparison_model_prefix_ids = comparison_model_prefix_ids.size()[-1]
            else:
                self.n_comparison_model_prefix_ids = 0
            if not starting_model_prefix_ids is None:
                self.n_starting_model_prefix_ids = starting_model_prefix_ids.size()[-1]
            else:
                self.n_starting_model_prefix_ids = 0

        def forward(self, **kwargs) -> CausalLMOutputWithPast:
            """
            kwargs will include
            - input_ids
            - attention_mask
            - past_key_values: (starting model, comparison model)
            - use cache
            - return_dict
            - output_attentions
            - output_hidden_states

            The comparison model should share all of them except past_key_values.
            """

            starting_model_input = kwargs.copy()
            comparison_model_input = kwargs.copy()
            if 'past_key_values' in kwargs and kwargs['past_key_values'] is not None:
                starting_model_input['past_key_values'] = kwargs['past_key_values'][0]
                comparison_model_input['past_key_values'] = kwargs['past_key_values'][1]
            #print(kwargs)
            # Apparently, kwargs doesn't preserve the additional arguments I put into the output, so this doesn't work:
            if ('starting_model_past_key_values' in kwargs and kwargs['starting_model_past_key_values'] is not None) and \
                ('comparison_model_past_key_values' in kwargs and kwargs['comparison_model_past_key_values'] is not None):
                
                print("starting_model_past_key_values", kwargs['starting_model_past_key_values'][0].size())
                print("comparison_model_past_key_values", kwargs['comparison_model_past_key_values'][0].size())

                starting_model_input['past_key_values'] = kwargs['starting_model_past_key_values']
                comparison_model_input['past_key_values'] = kwargs['comparison_model_past_key_values']
            else:
                if not self.comparison_model_prefix_ids is None and not self.comparison_model_prefix_ids_added:
                    prepended_ids = self.comparison_model_prefix_ids.repeat(comparison_model_input['input_ids'].size()[0], 1)
                    comparison_model_input['input_ids'] = torch.cat((prepended_ids, comparison_model_input['input_ids']), dim=1)
                    #self.comparison_model_prefix_ids_added = True
                comparison_model_input['input_ids'] = comparison_model_input['input_ids'].to(self.comparison_lm.device)

                if not self.starting_model_prefix_ids is None and not self.starting_model_prefix_ids_added:
                    prepended_ids = self.starting_model_prefix_ids.repeat(starting_model_input['input_ids'].size()[0], 1)
                    starting_model_input['input_ids'] = torch.cat((prepended_ids, starting_model_input['input_ids']), dim=1)
                    #self.starting_model_prefix_ids_added = True
                #print("starting_model_input['input_ids']", starting_model_input['input_ids'].size())
            starting_input_ids = starting_model_input['input_ids']
            comparison_input_ids = comparison_model_input['input_ids']
            
            # Instead, we use self.past_attn_storage to store past attention outputs, indexed by input_ids

            #print('\n\n', starting_model_input_key_str, starting_model_input['input_ids'], "self.past_starting_attn_storage.keys()", self.past_starting_attn_storage.keys())
            if self.cache_attn:
                starting_model_input_key_str = str(starting_model_input['input_ids'][:, :-1].tolist())
                if starting_model_input_key_str in self.past_starting_attn_storage:
                    starting_model_input['past_key_values'] = self.past_starting_attn_storage[starting_model_input_key_str]
                    starting_model_input['input_ids'] = starting_model_input['input_ids'][:, -1:]

                    del self.past_starting_attn_storage[starting_model_input_key_str]
                
                comparison_model_input_key_str = str(comparison_model_input['input_ids'][:, :-1].tolist())
                if comparison_model_input_key_str in self.past_comparison_attn_storage:
                    comparison_model_input['past_key_values'] = self.past_comparison_attn_storage[comparison_model_input_key_str]
                    comparison_model_input['input_ids'] = comparison_model_input['input_ids'][:, -1:]

                    del self.past_comparison_attn_storage[comparison_model_input_key_str]
            
            starting_model_output = super().forward(**starting_model_input)

            starting_model_probs = F.softmax(starting_model_output.logits, -1)
            starting_model_next_token_probs = starting_model_probs[:, -1, :]
            
            comparison_model_output = self.comparison_lm(**comparison_model_input)
            comparison_model_probs = F.softmax(comparison_model_output.logits, -1)
            if comparison_model_probs.size()[-1] > starting_model_probs.size()[-1]:
                comparison_model_probs = comparison_model_probs[:, :, :starting_model_probs.size()[-1]]
            comparison_model_next_token_probs = comparison_model_probs[:, -1, :]

            if self.cache_attn:
                new_starting_model_input_key_str = str(starting_input_ids.tolist())
                past_starting_keys = tuple(tuple(t.clone().detach() for t in layer_tuple) for layer_tuple in starting_model_output.past_key_values)
                self.past_starting_attn_storage[new_starting_model_input_key_str] = past_starting_keys
                
                new_comparison_model_input_key_str = str(comparison_input_ids.tolist())
                past_comparison_keys = tuple(tuple(t.clone().detach() for t in layer_tuple) for layer_tuple in comparison_model_output.past_key_values)
                self.past_comparison_attn_storage[new_comparison_model_input_key_str] = past_comparison_keys
            
            #print("starting_model_probs.size()", starting_model_probs.size(), "comparison_model_probs.size()", comparison_model_probs.size())
            #print("self.n_starting_model_prefix_ids", self.n_starting_model_prefix_ids, "self.n_comparison_model_prefix_ids", self.n_comparison_model_prefix_ids)
            comparison_model_probs = comparison_model_probs.to(starting_model_probs.device)
            subtract_prob = self.starting_model_weight * starting_model_probs[:, self.n_starting_model_prefix_ids:, :] + \
                            self.comparison_model_weight * comparison_model_probs[:, self.n_comparison_model_prefix_ids:, :]

            if self.similarity_gating_intensity is not None:
                similarity = torch.nn.functional.cosine_similarity(comparison_model_next_token_probs, starting_model_next_token_probs, dim=1)
                starting_model_bias = torch.exp(similarity * self.similarity_gating_intensity - self.similarity_gating_intensity)
                starting_model_bias = starting_model_bias.unsqueeze(1)
                #print(similarity, starting_model_bias)
                #print("starting_model_bias.size()", starting_model_bias.size())
                #print("subtract_prob[:, -1, :].size()", subtract_prob[:, -1, :].size())
                #print("starting_model_next_token_probs.size()", starting_model_next_token_probs.size())
                subtract_prob[:, -1, :] = (1 - starting_model_bias) * subtract_prob[:, -1, :] + starting_model_bias * starting_model_next_token_probs


            subtract_prob[subtract_prob < 0] = 0
            subtract_prob = subtract_prob + 1e-7
            new_logits = subtract_prob.log() # No need to normalize because this is the logit

            vocab_size = starting_model_probs.size()[-1]
            batch_size = starting_model_probs.size()[0]
            if self.limit_to_starting_model_top_p is not None:
                ordered_vocab_probs, ordered_prob_indices = torch.topk(starting_model_next_token_probs, k=vocab_size, dim=1)
                ordered_cumulative_vocab_probs = torch.cumsum(ordered_vocab_probs, dim=1)
                ordered_cumulative_vocab_probs_reached_p = ordered_cumulative_vocab_probs < self.limit_to_starting_model_top_p
                k_to_reach_p = torch.minimum(torch.sum(ordered_cumulative_vocab_probs_reached_p, dim=1) + 1, torch.tensor(vocab_size))

                logits_mask = -1000000 * torch.ones_like(ordered_vocab_probs)
                for i in range(batch_size):
                    valid_vocab_indices = ordered_prob_indices[i, :k_to_reach_p[i]]
                    #print("(i, valid_vocab_indices)", i, valid_vocab_indices)
                    logits_mask[i, valid_vocab_indices] = 0

                new_logits[:, -1, :] += logits_mask


            output = CausalLMOutputWithPast(
                loss=(starting_model_output.loss, comparison_model_output.loss),
                logits=new_logits,
                past_key_values=None, #\\\ (starting_model_output.past_key_values, comparison_model_output.past_key_values),
                hidden_states=(starting_model_output.hidden_states, comparison_model_output.hidden_states),
                attentions=(starting_model_output.attentions, comparison_model_output.attentions),
            )
            output['starting_model_logits'] = starting_model_output.logits[:, self.n_starting_model_prefix_ids:, :]
            output['comparison_model_logits'] = comparison_model_output.logits[:, self.n_comparison_model_prefix_ids:, :]
            output['starting_model_past_key_values'] = starting_model_output.past_key_values
            output['comparison_model_past_key_values'] = comparison_model_output.past_key_values
            return output

        
        def calculate_current_divergence(self, 
                                         text_ids : torch.Tensor, 
                                         batch_size : int = 16, 
                                         end_tokens_to_only_consider : int = 0,
                                         return_perplexities : bool = False,
                                         return_all_token_divergences : bool = False,
                                         logits_comparison_top_p : Optional[float] = None,
                                         use_avg_KL_as_divergences : bool = True,
                                         KL_distribution_choice : str = "auto"
                                        ) -> dict:
            n_texts = len(text_ids)
            n_batches = n_texts // batch_size + (n_texts % batch_size != 0)
            divergences = []
            all_token_divergences = []
            starting_model_perplexities = []
            comparison_model_perplexities = []
            for i in range(n_batches):
                batch_start = i * batch_size
                batch_end = min((i + 1) * batch_size, n_texts)
                current_ids = text_ids[batch_start : batch_end]
                output = self(**{"input_ids": current_ids, "labels": current_ids})
                starting_model_logits = output['starting_model_logits']
                comparison_model_logits = output['comparison_model_logits']
    
                vocab_size = starting_model_logits.size()[-1]
                n_tokens = starting_model_logits.size()[-2]

                if comparison_model_logits.size()[-1] > starting_model_logits.size()[-1]:
                    comparison_model_logits = comparison_model_logits[:, :, : vocab_size]

                div_starting_model_logits = starting_model_logits.clone()
                div_comparison_model_logits = comparison_model_logits.clone()
                
                # TODO:
                # if not logits_comparison_top_p is None:
                    # For each token in comparison_model_logits:
                    #     - compute vocab probabilities via softmax over the logits
                    #     - find the smallest n such that the total probability of the n most probable vocab entries is at least logits_comparison_top_p
                    #     - select only the top n vocab logits from comparison_model_logits
                # ^ Also need to change calculations of divergence so they can handle different vocab sizes for each token

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
                else:
                    loss_fct = torch.nn.L1Loss()
                    # Computes one divergence value per id sequence in the batch (output is of size batch_size):
                    batch_divergences = [loss_fct(div_comparison_model_logits[i,:,:], div_starting_model_logits[i,:,:]).item() for i in range(len(div_comparison_model_logits))]
                    if return_all_token_divergences:
                        # Computes one divergence value for every token in every sequence in the batch (output is of size batch_size x sequence_length):
                        token_divergences = [[loss_fct(div_comparison_model_logits[i,j,:], div_starting_model_logits[i,j,:]).item() for j in range(div_comparison_model_logits.size()[1])] for i in range(len(div_comparison_model_logits))]
                
                divergences = divergences + batch_divergences
                
                if return_all_token_divergences:
                    all_token_divergences = all_token_divergences + token_divergences

                if return_perplexities:
                    starting_model_batch_losses = torch.tensor([LM_loss(ids, logits, vocab_size).item() for logits,ids in zip(starting_model_logits, current_ids)])
                    comparison_model_batch_losses = torch.tensor([LM_loss(ids, logits, vocab_size).item() for logits,ids in zip(comparison_model_logits, current_ids)])
                    if end_tokens_to_only_consider > 0:
                        starting_model_batch_losses = starting_model_batch_losses * (n_tokens / (end_tokens_to_only_consider))
                        comparison_model_batch_losses = comparison_model_batch_losses * (n_tokens / (end_tokens_to_only_consider))

                    starting_model_batch_perplexities = torch.exp(starting_model_batch_losses).tolist()
                    comparison_model_batch_perplexities = torch.exp(comparison_model_batch_losses).tolist()

                    starting_model_perplexities = starting_model_perplexities + starting_model_batch_perplexities
                    comparison_model_perplexities = comparison_model_perplexities + comparison_model_batch_perplexities

            result = {'divergences': divergences}
            if return_perplexities:
                result['starting_model_perplexities'] = starting_model_perplexities
                result['comparison_model_perplexities'] = comparison_model_perplexities
            if return_all_token_divergences:
                result['all_token_divergences'] = all_token_divergences
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


def get_cls(model_name : str) -> Type[PreTrainedModel]:
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
    else:
        raise ValueError("Model name not recognized.")

def instantiate_models(
        model_name : str = "gpt2-xl",
        starting_model_path : str = "gpt2-xl",
        comparison_model_path : str = "gpt2-xl",
        starting_model_weight : float = -1,
        comparison_model_weight : float = 1,
        tokenizer_family : str = "gpt2",
        device : str = "cuda:0",
        temp_save_model_loc : str = "/tmp/temp_",
        limit_to_starting_model_top_p : Optional[float] = None,
        similarity_gating_intensity : Optional[float] = None,
        comparison_model_prefix_ids : Optional[List[int]] = None,
        starting_model_prefix_ids : Optional[List[int]] = None,
        use_4_bit : bool = True,
        no_quantize_base_model : bool = False,
        bnb_config : Optional[dict] = None,
        cache_attn : bool = False
        ) -> Tuple[PreTrainedModel, PreTrainedModel, PreTrainedModel, PreTrainedTokenizer]:
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
        starting_model = None

    if ".pth" in comparison_model_path:
        comparison_model = torch.load(comparison_model_path)#.to(device)
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
        comparison_model = AutoModelForCausalLM.from_pretrained(comparison_model_path, 
                                                                load_in_4bit=use_4_bit, 
                                                                device_map={"": 0} if device == "cuda:0" else "auto",
                                                                quantization_config=bnb_config)#.to(device)

    comparison_model = comparison_model.eval()
    model_class = get_cls(model_name)
    set_verbosity_error()
    model = build_contrastive_lm(model_class).from_pretrained(
        starting_model_path,
        starting_model_weight=starting_model_weight, 
        comparison_model_weight=comparison_model_weight,
        limit_to_starting_model_top_p=limit_to_starting_model_top_p,
        similarity_gating_intensity=similarity_gating_intensity,
        comparison_model_prefix_ids=comparison_model_prefix_ids,
        starting_model_prefix_ids=starting_model_prefix_ids,
        quantization_config=bnb_config if not no_quantize_base_model else None,
        load_in_4bit=use_4_bit and not no_quantize_base_model, 
        device_map={"": 0} if device == "cuda:0" else "auto",
        bnb_config=bnb_config if not no_quantize_base_model else None,
        cache_attn=cache_attn,
    ).eval()#.to(device)
    #print(model)
    #for name, param in model.named_parameters():
    #    print('name:', name, 'precision:', param.dtype)
    model.comparison_lm = comparison_model

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_family)
    #print(tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    
    return model, starting_model, comparison_model, tokenizer

def get_input_ids(
        tokenizer : PreTrainedTokenizer,
        single_prefix : Optional[str] = None,
        text_set : Optional[List[str]] = None,
        prefixes_path : Optional[str] = None,
        set_prefix_len : Optional[int] = None,
        n_prefixes : Optional[int] = None,
        device : str = "cuda:0"
        ) -> torch.Tensor:
    if not single_prefix is None:
        prompt = [single_prefix]
        if not n_prefixes is None:
            prompt = [single_prefix] * n_prefixes
    elif not prefixes_path is None:
        if ".txt" in prefixes_path:
            prompt = open(prefixes_path, "r").readlines()
            prompt = [p.replace("\n", "") for p in prompt]
        elif ".csv" in prefixes_path:
            prompt = read_csv(prefixes_path).values[:,1].tolist()
        prompt = [s for s in prompt if not '<unk>' in s]
    elif not text_set is None:
        prompt = text_set
    else:
        return None
    input_ids = tokenizer.batch_encode_plus(prompt, padding=True, truncation=True, return_tensors="pt")['input_ids'].to(device)
    if not set_prefix_len is None:
        input_ids = input_ids[:, :set_prefix_len]
        # Filter out any entry in input_ids that has padding
        if not "gpt" in str(type(tokenizer)).lower():
            input_ids = input_ids[~(input_ids.eq(tokenizer.pad_token_id)).any(dim=1)]
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


