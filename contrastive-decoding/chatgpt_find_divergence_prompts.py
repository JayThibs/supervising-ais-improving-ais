import torch
from model_comparison_helpers import instantiate_models
import tqdm
import json
from openai import OpenAI
import re
from random import sample
from contrastive_decoding import ContrastiveDecoder
import numpy as np
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from terminaltables import AsciiTable
import textwrap


class DivergenceFinder:
    def __init__(self, **kwargs):
        # Default values
        self.model_name = "gpt2-xl"
        self.generation_length = 20
        self.n_cycles_ask_chatgpt = 2
        self.openai_model_str = None
        self.local_model_str = "Upstage/SOLAR-10.7B-Instruct-v1.0"
        self.generations_per_prefix = 3
        self.starting_model_path = "gpt2-xl"
        self.comparison_model_path = "gpt2-xl"
        self.starting_model_weight = 1
        self.comparison_model_weight = -1
        self.tokenizer_family = "gpt2"
        self.set_prefix_len = 30
        self.device = "cuda:0"
        self.local_model_device = "cuda:0"
        self.temp_save_model_loc = "/tmp/temp_"
        self.limit_to_starting_model_top_p = -1
        self.similarity_gating_intensity = -1
        self.comparison_model_prefix_ids = None
        self.starting_model_prefix_ids = None
        self.sequential = True
        self.n_past_texts_subsampled = 10
        self.subsampling_randomness_temperature = 0.5
        self.api_key_path = "../key.txt"
        self.prompts_json_path = "chatgpt_prompts/who_is_harry_potter_find_CD_prompts.json"
        self.quantize = True
        self.divergence_fnct = 'l1'
        self.include_prefix_in_divergences = False
        self.verbose = True
        self.max_width = 70
        self.use_custom_selection_criterion = False
        self.use_custom_selection_criterion_examples = False

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
                                                                    device_map=local_device_map,
                                                                    quantization_config=self.bnb_config)
            self.local_tokenizer = AutoTokenizer.from_pretrained(self.local_model_str)
        elif self.openai_model_str is not None and self.local_model_str is not None:
            raise ValueError("You cannot specify both openai_model_str and local_model_str.")
        elif self.openai_model_str is None and self.local_model_str is None:
            raise ValueError("You must specify either openai_model_str or local_model_str.")
        
        if not self.quantize:
            self.bnb_config = None
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
            bnb_config = self.bnb_config,
        )
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
            "starting_model": self.starting_model,
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
            "include_prefix_in_divergences": self.include_prefix_in_divergences,
            "quantize": self.quantize,
            "divergence_fnct": self.divergence_fnct
        }
        self.contrastive_decoder = ContrastiveDecoder(**contrastive_decoder_params)

        if len(self.seed_demonstrations_list) > 0:
            # Compute divergences for seed_demonstrations_list
            result = self.contrastive_decoder.decode()
            divergences = result['divergences'].tolist()

            self.all_divergences_and_texts = [(d,s,"", 0) for d,s in zip(divergences, self.seed_demonstrations_list)]
            self.round_divergences_and_texts = divergence_weighted_subsample(self.all_divergences_and_texts, self.n_past_texts_subsampled, self.subsampling_randomness_temperature)
            if self.verbose:
                print("round_divergences_and_texts:")
                print_texts_with_divergences(self.round_divergences_and_texts, max_width = self.max_width)
        else:
            self.all_divergences_and_texts = []
            self.round_divergences_and_texts = []

        # Read OpenAI auth key from a file
        with open(self.api_key_path, 'r') as file:
            openai_auth_key = file.read().strip()

        # Authenticate with the OpenAI API
        self.client = OpenAI(api_key=openai_auth_key)

    def get_continuation(self, messages, openai_model_str, client, local_model, local_tokenizer, device, verbose_continuations = False):
        if openai_model_str is not None:
            # Generate a new text from ChatGPT
            if verbose_continuations:
                print("messages:", messages)
            chat_completion = client.chat.completions.create(
                model=openai_model_str,
                messages=messages,
                max_tokens=2000,
                temperature=1
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
            with torch.no_grad():
                output = local_model.generate(
                    input_ids=prompt_tokens,
                    max_length=2000,
                    temperature=1,
                    do_sample=True,
                    top_p=0.9,
                    top_k=0,
                    num_return_sequences=1,
                    return_dict_in_generate=True
                )
            output_tokens = output.sequences[0][n_prompt_tokens:]
            output_text = local_tokenizer.decode(output_tokens, skip_special_tokens=True)

        return output_text
    
    # This function takes in a list of prompts and responses, and returns a list of scores
    # where each score either 0 or 1, indicating whether each prompt satisfied the custom 
    # selection criterion.
    def score_by_custom_selection_criterion(self, prompts_and_responses, openai_model_str, client, local_model, local_tokenizer, custom_selection_criterion, yes_response_str, no_response_str, custom_selection_criterion_examples, device):
        scores = []
        #print(prompts_and_responses)
        for _, prompt, response, _ in prompts_and_responses:
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
                device=device
            )
            # Determine if the assistant's response matches the expected "yes" or "no" responses
            if assistant_response.split(" ")[0].strip().rstrip('.!?') in yes_response_str:
                scores.append(1)
            elif assistant_response.split(" ")[0].strip().rstrip('.!?') in no_response_str:
                scores.append(0)
            else:
                # If the response is neither "yes" nor "no", log an error and append a default score
                print(f"Error: Unexpected response from assistant model: {assistant_response}")
                scores.append(0)  # Default score for unexpected responses
        return scores
    
    def search_step(self):
        if self.verbose:
            print("round_divergences_and_texts:")
            print_texts_with_divergences(self.round_divergences_and_texts, max_width = self.max_width)
        # Compose messages to send ChatGPT
        if len(self.round_divergences_and_texts) > 0:
            # First, rescale round_divergences_and_texts divergence values into [0, 10]
            rescaled_round_divergences_and_texts = rescale_divergences(self.round_divergences_and_texts)
            # Format divergences and texts into a single string to provide ChatGPT
            current_divergences_and_texts_str = "\n".join([f"divergence {i}: {round(div, 3)}, input text {i}: {text}" for i, (div, text) in enumerate(rescaled_round_divergences_and_texts)])
            messages=[
            {"role": "system", "content": self.system_start_prompt + "\n" + self.system_loop_prompt},
            {"role": "user", "content": current_divergences_and_texts_str + "\n" + self.user_end_prompt}
            ]

        else:
             messages=[{"role": "system", "content": self.system_start_prompt},
            {"role": "user", "content": self.user_end_prompt}
             ]

        # Get new messages from either ChatGPT or a local model:
        output_text = self.get_continuation(messages, self.openai_model_str, self.client, self.local_model, self.local_tokenizer, self.device)
        #print("output_text:", output_text)
        additional_texts = interpret_assistant_outputs(output_text)
        print("additional_texts:", additional_texts)

        if len(additional_texts) > 0:
            # Compute divergences for additional_texts
            result = self.contrastive_decoder.decode(text_set = additional_texts)
            additional_divergences = result['divergences'].tolist()
            comparison_model_responses = []
            with torch.no_grad():
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
                    comparison_model_responses.append(decoded_output)
            custom_scores = [0 for _ in additional_divergences]
            additional_divergences_and_texts = list(zip(additional_divergences, additional_texts, comparison_model_responses, custom_scores))
            if self.use_custom_selection_criterion:
                custom_scores = self.score_by_custom_selection_criterion(additional_divergences_and_texts, self.openai_model_str, self.client, self.local_model, self.local_tokenizer, self.custom_selection_criterion, self.custom_selection_criterion_yes_response, self.custom_selection_criterion_no_response, self.custom_selection_criterion_examples, self.device)
                if self.verbose:
                    table_data = [["#", "Prompt", "Response", "Custom Score Boolean"]]
                    for i, (_, prompt, response, custom_score) in enumerate(zip(additional_divergences, additional_texts, comparison_model_responses, custom_scores)):
                        custom_score_boolean = custom_score != 0
                        row = [str(i+1), textwrap.fill(prompt.replace("\n", "\\n"), width=self.max_width), textwrap.fill(response.replace("\n", "\\n"), width=self.max_width), str(custom_score_boolean)]
                        table_data.append(row)
                    table = AsciiTable(table_data)
                    table.inner_row_border = True
                    print(table.table)
                additional_divergences_and_texts = list(zip(additional_divergences, additional_texts, comparison_model_responses, custom_scores))
            # Expand all_divergences_and_texts with results:
            self.all_divergences_and_texts = self.all_divergences_and_texts + additional_divergences_and_texts
        # Assemble a new query from ChatGPT's responses / additional_texts.
        if self.sequential and len(additional_texts) > 0:
            # Feed ChatGPT only its own responses from this round, from additional_divergences and additional_texts
            self.round_divergences_and_texts = additional_divergences_and_texts
        else:
            # If not sequential (or if we didn't get any additional texts from this round), 
            # then feed ChatGPT responses from all_divergences_and_texts
            if len(additional_texts) == 0:
                print("Warning: No additional texts were obtained from this round.")
            # Subsample from all_divergences_and_texts
            
            self.round_divergences_and_texts = divergence_weighted_subsample(self.all_divergences_and_texts, self.n_past_texts_subsampled, self.subsampling_randomness_temperature, self.use_custom_selection_criterion)
            
            if self.verbose:
                print("round_divergences_and_texts:")
                print_texts_with_divergences(self.round_divergences_and_texts, max_width = self.max_width)
    
    def search_loop(self):
        for _ in tqdm.tqdm(range(self.n_cycles_ask_chatgpt)):
            self.search_step()
        try:
            # Sort the list by divergence values in descending order
            self.all_divergences_and_texts.sort(key=lambda x: x[0], reverse=True)
            self.all_divergences_and_texts.sort(key=lambda x: x[0], reverse=True)

            # Print the 10 texts with the highest divergence values
            table_data = [["#", "Divergence", "Text", "Response", "CSB"]]
            for i in range(min(10, len(self.all_divergences_and_texts))):
                wrapped_text = textwrap.fill(self.all_divergences_and_texts[i][1].replace("\n", "\\n"), width=self.max_width)
                wrapped_response = textwrap.fill(self.all_divergences_and_texts[i][2].replace("\n", "\\n"), width=self.max_width)
                row = [str(i+1), str(round(self.all_divergences_and_texts[i][0], 3)), wrapped_text, wrapped_response, str(self.all_divergences_and_texts[i][3] != 0)]
                table_data.append(row)
            table = AsciiTable(table_data)
            table.inner_row_border = True
            print(table.table)
            
            # Print the 10 texts with the lowest divergence values
            table_data_lowest = [["#", "Divergence", "Text", "Response", "CSB"]]
            for i in range(min(10, len(self.all_divergences_and_texts))):
                wrapped_text = textwrap.fill(self.all_divergences_and_texts[-(i+1)][1].replace("\n", "\\n"), width=self.max_width)
                wrapped_response = textwrap.fill(self.all_divergences_and_texts[-(i+1)][2].replace("\n", "\\n"), width=self.max_width)
                row = [str(i+1), str(round(self.all_divergences_and_texts[-(i+1)][0], 3)), wrapped_text, wrapped_response, str(self.all_divergences_and_texts[-(i+1)][3] != 0)]
                table_data_lowest.append(row)
            table_lowest = AsciiTable(table_data_lowest)
            table_lowest.inner_row_border = True
            print(table_lowest.table)
        except:
            print("Error: Could not print texts with highest/lowest divergence values.")

    def find_diverging_texts(self, **kwargs):
        if kwargs:
            for key, value in kwargs.items():
                setattr(self, key, value)
        self.search_loop()
        return self.all_divergences_and_texts
    

# This function takes in a list of divergences and texts, and rescales the divergences into [0, 10].
# It returns a list of pairs of divergences and texts, where the texts are the same as the input texts.
#
# divergences_and_texts: A list of pairs of divergences and texts.
def rescale_divergences(divergences_and_texts):
    max_divergence = max(divergences_and_texts, key=lambda x: x[0])[0]
    min_divergence = min(divergences_and_texts, key=lambda x: x[0])[0]
    rescaled_round_divergences_and_texts = [(10 * (div - min_divergence) / max((max_divergence - min_divergence), 0.0001), text) for div, text, _, _ in divergences_and_texts]
    return rescaled_round_divergences_and_texts


# This function takes in a string of text and interprets it as a json object, returning a list of texts.
# If the json parsing fails, it tries to interpret the text as a list of texts separated by newlines, and returns a list of texts.
#
# output_text: The string of text to interpret.
# as_json: Whether to interpret the text as a json object or a newline-separated list of texts.
# fallback_to_newlines: Whether to interpret the text as a newline-separated list of texts if the json parsing fails.
def interpret_assistant_outputs(output_text, as_json = True, fallback_to_newlines = False):
    json_parse_fail = False
    if as_json:
        # Interpret output_text as a json object with a dictionary and parse it:
        # First, remove all characters before the first '{' character
        output_text = output_text[output_text.find('{'):]
        # Next, remove all characters after the last '}' character
        output_text = output_text[:output_text.rfind('}')+1]
        try:
            #print("json text", output_text)
            output_text_json = json.loads(output_text)
        except:
            print("Error: Could not parse output_text as a json object.")
            print(output_text)
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
    

# This function takes in a list of divergences and texts, and returns a list of pairs of divergences and texts, where 
# the texts are a random subsample of the input texts, weighted by the divergence values, with subsampling_randomness_temperature
# scaling the degree of randomness.
# Optionally takes in a boolean to use scores from a custom selection criterion to bias selection process.
def divergence_weighted_subsample(divergence_and_texts, n_past_texts_subsampled = 5, subsampling_randomness_temperature = 0.5, select_high_divergence = False, use_custom_selection_criterion = False):
    if use_custom_selection_criterion:
        weights = [entry[0] + entry[3] for entry in divergence_and_texts]
    else:
        weights = [entry[0] for entry in divergence_and_texts]

    if subsampling_randomness_temperature > 0:
        if select_high_divergence:
            weights = np.exp(np.array(weights) / subsampling_randomness_temperature)
        else:
            weights = np.exp(-np.array(weights) / subsampling_randomness_temperature)
        weights /= np.sum(weights)

        # Randomly choose indices based on weights
        chosen_indices = np.random.choice(len(divergence_and_texts), size=n_past_texts_subsampled, p=weights)
    else:
        # If no randomness, just select the top n_past_texts_subsampled texts by weights
        chosen_indices = np.argsort(weights)[-n_past_texts_subsampled:]

    # Select the subsampled pairs
    subsample = [divergence_and_texts[i] for i in chosen_indices]

    # Sort the subsample by divergence
    subsample = sorted(subsample, key=lambda x: x[0], reverse=True)

    # Return the subsample
    return subsample

def print_texts_with_divergences(divergences_and_texts, max_width = 50):
    table_data = [["Div", "Text", "Response", "CSB"]]
    for div, text, response, custom_score in divergences_and_texts:
            wrapped_text = "\n".join(textwrap.wrap(text, width=max_width))
            wrapped_response = "\n".join(textwrap.wrap(response, width=max_width))
            table_data.append([round(div, 3), wrapped_text, wrapped_response, str(custom_score != 0)])
    table = AsciiTable(table_data)
    table.inner_row_border = True
    print(table.table)

