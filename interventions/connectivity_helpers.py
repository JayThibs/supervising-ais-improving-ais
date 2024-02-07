import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM
from eval_helpers import *
import tqdm

def linear_model_mix(model_1, model_2, model_2_weight, output_model, device):
    for p_om, p_1, p_2 in zip(output_model.parameters(), model_1.parameters(), model_2.parameters()):
        p_om.data = (1 - model_2_weight) * p_1.data + model_2_weight * p_2.data
    return output_model

def check_barrier_heights(model_1, model_2, data, labels, batch_size = 32, n_check_points_between = 49, device = "cuda:0", pad_token_id = 0, checking_generator = False):
    if checking_generator:
        output_model =  AutoModelForCausalLM.from_config(model_1.config).to(device)
    else:
        output_model = AutoModelForSequenceClassification.from_config(model_1.config).to(device)
    path_losses = []
    for i in tqdm.tqdm(range(0, n_check_points_between + 2)):
        prior_model_weight = i / (n_check_points_between + 2)
        linear_interpolation_model = linear_model_mix(model_1, model_2, prior_model_weight, output_model, device = device)

        linear_interpolation_model_loss, linear_interpolation_model_accuracy = eval_model(linear_interpolation_model, data, labels, batch_size, pad_token_id)
        path_losses.append(linear_interpolation_model_loss)
    return path_losses

def find_tallest_barrier(model_1, model_2, data, labels, batch_size = 32, n_check_points_between = 5):
    pass
    # TODO: Write method of finding tall barriers by using local gradients
