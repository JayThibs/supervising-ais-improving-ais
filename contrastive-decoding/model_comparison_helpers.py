import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel, OPTForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.nn.functional as F


# Adapted from: https://github.com/xiamengzhou/training_trajectory_analysis/blob/main/utils.py
class CausalLMSubtract(GPT2LMHeadModel):
    def __init__(self,
                 config,
                 comparison_lm,
                 starting_model_weight=-1,
                 comparison_model_weight=1,
                 limit_to_starting_model_top_p=-1,
                 similarity_gating_intensity=-1):
        super().__init__(config)
        self.comparison_lm = GPT2LMHeadModel.from_pretrained(comparison_lm)
        self.starting_model_weight = starting_model_weight
        self.comparison_model_weight = comparison_model_weight

        self.similarity_gating_intensity = similarity_gating_intensity
        self.limit_to_starting_model_top_p = limit_to_starting_model_top_p

    def forward(self, **kwargs):
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

        starting_model_output = super().forward(**starting_model_input)
        starting_model_probs = F.softmax(starting_model_output.logits, -1)
        starting_model_next_token_probs = starting_model_probs[:, -1, :]

        comparison_model_output = self.comparison_lm(**comparison_model_input)
        comparison_model_probs = F.softmax(comparison_model_output.logits, -1)
        comparison_model_next_token_probs = comparison_model_probs[:, -1, :]

        subtract_prob = self.starting_model_weight * starting_model_probs + self.comparison_model_weight * comparison_model_probs

        if self.similarity_gating_intensity != -1:
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
        if self.limit_to_starting_model_top_p != -1:
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
            past_key_values=None, # (starting_model_output.past_key_values, comparison_model_output.past_key_values),
            hidden_states=(starting_model_output.hidden_states, comparison_model_output.hidden_states),
            attentions=(starting_model_output.attentions, comparison_model_output.attentions),
        )
        output['model_1_logits'] = starting_model_output.logits
        output['model_2_logits'] = comparison_model_output.logits
        return output


def estimate_cont_divergence(clms, text, n_estimations = 1, generation_length = 20):
    # TODO
    pass
