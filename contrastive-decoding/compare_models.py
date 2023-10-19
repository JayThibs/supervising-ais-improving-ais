import torch
import transformers
from transformers import AutoModel, AutoModelForCausalLM, GPT2LMHeadModel, OPTForCausalLM, AutoTokenizer
import pandas as pd
import argparse
import os
from model_comparison_helpers import CausalLMSubtract
import numpy as np
import shutil

 
parser = argparse.ArgumentParser(description='Model training script for linear basin connectivity experiments.')

parser.add_argument('--model_name', type=str, default='gpt2',
                    help='Huggingface model name.')

parser.add_argument('--generation_length', type=int, default=20,
                    help='How many additional tokens to generate after each prefix.')

parser.add_argument('--generations_per_prefix', type=int, default=1,
                    help='How many texts to generate for each prefix.')

parser.add_argument('--model_1_path', type=str, default=None,
                    help='Location of first model.')

parser.add_argument('--model_2_path', type=str, default=None,
                    help='Location of second model.')

parser.add_argument('--model_1_weight', type=float, default=1,
                    help='Weight of model 1 in summed token probs.')

parser.add_argument('--model_2_weight', type=float, default=-1,
                    help='Weight of model 2 in summed token probs.')

parser.add_argument('--tokenizer_family', type=str, default="gpt2",
                    help='Name of a model that uses the appropriate HuggingFace tokenizer for the models being used.')

parser.add_argument('--single_prefix', type=str, default=None,
                    help='Prefix to use for all generations.')

parser.add_argument('--prefixes_path', type=str, default=None,
                    help='Location of file containing prompting prefixes (one per line).')

#parser.add_argument('--models_dir_path', type=str, default=None,
#                    help='Location of models directory. If given, we check connectivity between each pair of models in the directory.')

#parser.add_argument('--batch_size', type=int, default=32,
#                    help='Batch size during training.')

parser.add_argument('--device', type=str, default="cuda:0",
                    help='GPU / CPU device on which experiments will run.')

parser.add_argument('--save_texts_loc', type=str, default='None',
                    help='Where do we save the generated texts? Keep set to \"None\" to disable.')

parser.add_argument('--temp_save_model_loc', type=str, default='/tmp/temp_',
                    help='Path we use to save a temp version of the model, for when we need to do that during model loading.')

parser.add_argument('--print_texts', type=str, default='true',
                    help='Do we print the generated texts? Any value but "true" means we do not.')

parser.add_argument('--sampling', type=str, default='true',
                    help='Should we use neucleus sampling (\'true\') or greedy (\'false\')?')

parser.add_argument('--top_p', type=float, default=0.95,
                    help='Top p for neucleus sampling.')

parser.add_argument('--limit_to_starting_model_top_p', type=float, default=-1,
                    help='Set to some value in [0, 1), and the allowed tokens for the contrastively decoded outputs will be restricted to only\
                          the set of most probable tokens whose total probability sums to no more than that value, according to the starting\
                          model\'s logits.')

parser.add_argument('--similarity_gating_intensity', type=float, default=-1,
                    help='Set to some value (~400 is suggested) to enable adaptive deference to the starting model\s probability distribution\
                          in the case that both the starting and comparison models have very similar probability distributions.\
                          1 / similarity_gating_intensity basically serves as the temperature of an exponential function of the cosine\
                          similarity between the two probability distributions, whose output determines how much we defer to the starting\
                          model. Higher similarity_gating_intensity => requiring greater similarity before we defer.')


args = parser.parse_args()


if ".pth" in args.model_1_path:
    model_1 = torch.load(args.model_1_path)
    model_1_name = args.model_1_path.split("/")[-1][:-4]
    model_1_temp_save_pretrained_dir = args.temp_save_model_loc + model_1_name
    try:
        shutil.rmtree(model_1_temp_save_pretrained_dir)
    except:
        pass
    os.mkdir(model_1_temp_save_pretrained_dir)
    model_1.save_pretrained(model_1_temp_save_pretrained_dir)
    args.model_1_path = model_1_temp_save_pretrained_dir
else:
    model_1 = AutoModelForCausalLM.from_pretrained(args.model_1_path)

if ".pth" in args.model_2_path:
    model_2 = torch.load(args.model_2_path)
    model_2_name = args.model_2_path.split("/")[-1][:-4]
    model_2_temp_save_pretrained_dir = args.temp_save_model_loc + model_2_name
    try:
        shutil.rmtree(model_2_temp_save_pretrained_dir)
    except:
        pass
    os.mkdir(model_2_temp_save_pretrained_dir)
    model_2.save_pretrained(model_2_temp_save_pretrained_dir)
    args.model_2_path = model_2_temp_save_pretrained_dir
else:
    model_2 = AutoModelForCausalLM.from_pretrained(args.model_2_path)


transformers.utils.logging.set_verbosity_error()
model = CausalLMSubtract.from_pretrained(
    args.model_1_path, 
    model_2=args.model_2_path, 
    model_1_weight=args.model_1_weight, 
    model_2_weight=args.model_2_weight,
    limit_to_starting_model_top_p=args.limit_to_starting_model_top_p,
    similarity_gating_intensity=args.similarity_gating_intensity,
).to(args.device)

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_family)
if 'gpt2' in str.lower(str(type(tokenizer))):
    tokenizer.pad_token = tokenizer.eos_token 


if not args.single_prefix is None:
    prompt = [args.single_prefix]
elif not args.prefixes_path is None:
    if ".txt" in args.prefixes_path:
        prompt = open(args.prefixes_path, "r").readlines()
        prompt = [p.replace("\n", "") for p in prompt]
    elif ".csv" in args.prefixes_path:
        prompt = pd.read_csv(args.prefixes_path).values[:,1].tolist()
input_ids = tokenizer.batch_encode_plus(prompt, padding=True, truncation=True, return_tensors="pt")['input_ids'].to(args.device)

# neucleus sampling (sampling=True):
# greedy search (sampling=False): 
sampling = str.lower(args.sampling) == 'true'
generations = []
for ids in input_ids:
    ids = ids[ids != tokenizer.pad_token_id]
    ids = torch.unsqueeze(ids, 0)
    generation = model.generate(ids, do_sample=sampling, max_new_tokens=args.generation_length, top_k=None, top_p=args.top_p, num_return_sequences=args.generations_per_prefix).tolist()
    generations += generation
generated_texts = tokenizer.batch_decode(generations)


if str.lower(args.print_texts) == 'true':
    for t in generated_texts:
        print(t)

if str.lower(args.save_texts_loc) != "none":
    (pd.DataFrame(generated_texts)).to_csv(args.save_texts_loc)

