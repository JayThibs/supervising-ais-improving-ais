import torch
import transformers
from transformers import AutoModel, AutoModelForSequenceClassification, BertForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt

from connectivity_helpers import *
from train_helpers import *
from get_data import *

parser = argparse.ArgumentParser(description='Model training script for linear basin connectivity experiments.')

parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                    help='Huggingface model name.')

parser.add_argument('--dataset', type=str, default='mnli',
                    help='Huggingface dataset name to train the model on. Can also be a path to a saved dataset.')

parser.add_argument('--load_saved_data', type=str, default='true',
                    help='Do we try to load a saved dataset, or re-download the data?')

parser.add_argument('--longest_sequence_allowed', type=int, default=64,
                    help='Ignore datapoints with more than this many tokens.')

parser.add_argument('--test_set_size', type=int, default=10000,
                    help='Number of datapoints to include in the test set.')

parser.add_argument('--use_test_data', type=str, default='true',
                    help='Set to any value but \"true\" in order to switch to using the last test_set_size datapoints from the training data instead.')

parser.add_argument('--make_binary', type=str, default='true',
                    help='Do we convert the datasets to a binary prediction problem? Any value but "true" means we do not.')

parser.add_argument('--model_1_path', type=str, default=None,
                    help='Location of first model.')

parser.add_argument('--model_2_path', type=str, default=None,
                    help='Location of second model.')

parser.add_argument('--models_dir_path', type=str, default=None,
                    help='Location of models directory. If given, we check connectivity between each pair of models in the directory.')

parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size during training.')

parser.add_argument('--n_interpolation_points', type=int, default=20,
                    help='Number of points to check on the linear path between the models. Total model evaluations are 2 higher than this number due to evaluating endpoints.')

parser.add_argument('--device', type=str, default="cuda:0",
                    help='GPU / CPU device on which experiments will run.')

parser.add_argument('--show_plot', type=str, default='true',
                    help='Do we show a plot of the interpolation losses? Any value but "true" means we do not.')

parser.add_argument('--save_plot_loc', type=str, default='None',
                    help='Where do we save a plot of the interpolation losses? Keep set to \"None\" to disable.')

parser.add_argument('--save_plot_ext', type=str, default='.pdf',
                    help='What file extension do we give our saved plots? Defaults to .pdf')

parser.add_argument('--save_losses_loc', type=str, default='None',
                    help='Where do we save a CSV of the interpolation losses? Keep set to \"None\" to disable.')

parser.add_argument('--print_losses', type=str, default='true',
                    help='Do we print a list of the interpolation losses? Any value but "true" means we do not.')

parser.add_argument('--checking_generator', type=str, default='false',
                    help='Do we init our model as a generator (causal) language model? Defaults to false')

args = parser.parse_args()


dataset_label = str.lower(args.dataset)
data_path = "data/" + dataset_label
make_binary = str.lower(args.make_binary) == "true"
checking_generator = str.lower(args.checking_generator) == 'true'

train_data, train_labels, test_data, test_labels, tokenizer = load_data(args.model_name, 
                                                                        dataset_label, 
                                                                        args.load_saved_data, 
                                                                        data_path, 
                                                                        args.longest_sequence_allowed, 
                                                                        args.test_set_size, 
                                                                        make_binary, 
                                                                        args.device)

if str.lower(args.use_test_data) != "true":
    print("Using training data to evaluate interpolation losses.")
    test_data = train_data[:args.test_set_size]
    test_labels = train_labels[:args.test_set_size]
else:
    test_data = test_data[:args.test_set_size]
    test_labels = test_labels[:args.test_set_size]

all_path_losses = []

if (not args.models_dir_path is None) and (args.model_1_path is None) and (args.model_2_path is None):
    if args.models_dir_path[-1] != "/":
        args.models_dir_path += "/"
    model_names = [name for name in os.listdir(args.models_dir_path) if name[-4:] == ".pth"]
    print("Checking all pairs connectivity among:")
    print(model_names)
    n_models = len(model_names)
    for i in range(n_models):
        model_1 = torch.load(args.models_dir_path + model_names[i])
        model_1.config.pad_token_id = tokenizer.pad_token_id
        for j in range(i, n_models):
            model_2 = torch.load(args.models_dir_path + model_names[j])
            model_2.config.pad_token_id = tokenizer.pad_token_id
            path_losses = check_barrier_heights(model_1, 
                                                model_2, 
                                                test_data, 
                                                test_labels, 
                                                batch_size = args.batch_size, 
                                                n_check_points_between = args.n_interpolation_points, 
                                                device = args.device, 
                                                pad_token_id = tokenizer.pad_token_id,
                                                checking_generator = checking_generator)

            all_path_losses.append([path_losses, model_names[i] + "__" + model_names[j]])

elif (args.models_dir_path is None) and (not args.model_1_path is None) and (not args.model_2_path is None):
    model_1 = torch.load(args.model_1_path)
    model_2 = torch.load(args.model_2_path)
    model_1_name = args.model_1_path.split("/")[-1]
    model_2_name = args.model_2_path.split("/")[-1]
    model_1.config.pad_token_id = tokenizer.pad_token_id
    model_2.config.pad_token_id = tokenizer.pad_token_id

    path_losses = check_barrier_heights(model_1, 
                                        model_2, 
                                        test_data, 
                                        test_labels, 
                                        batch_size = args.batch_size, 
                                        n_check_points_between = args.n_interpolation_points, 
                                        device = args.device, 
                                        checking_generator = checking_generator)

    all_path_losses.append([path_losses, model_1_name + "__" + model_2_name])

if str.lower(args.save_plot_loc) != 'none' and args.save_plot_loc[-1] != '/':
    args.save_plot_loc += '/'
if str.lower(args.save_losses_loc) != "none" and args.save_losses_loc[-1] != '/':
    args.save_losses_loc += '/'

for plot_result in all_path_losses:
    path_losses = plot_result[0]
    pair_name = plot_result[1]
    if str.lower(args.show_plot) == "true" or str.lower(args.save_plot_loc) != "none":
        plt.plot(range(args.n_interpolation_points + 2), path_losses)
        plt.xlabel("Interpolation point number")
        plt.ylabel(("Train" if str.lower(args.use_test_data) != "true" else "Test") + " Loss")
        if str.lower(args.show_plot) == "true":
            plt.show()
        if str.lower(args.save_plot_loc) != "none":
            plt.savefig(args.save_plot_loc + "connectivity_pair_plot_" + pair_name + args.save_plot_ext)
    
    if str.lower(args.save_losses_loc) != "none":
        (pd.DataFrame(path_losses)).to_csv(args.save_losses_loc + "connectivity_pair_losses_" + pair_name + ".csv")

    if str.lower(args.print_losses) == "true":
        print(path_losses)
    plt.cla()
