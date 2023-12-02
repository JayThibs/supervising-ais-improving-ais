import torch
import transformers
from transformers import AutoTokenizer
from datasets import load_dataset
import pandas as pd
from convo_tree_helpers import extract_convo_trees


def get_mnli_dataset(model_name = "bert-base-uncased", 
                     longest_sequence_allowed = 64, 
                     n_test_data = 10000, 
                     make_binary = True, 
                     data_path = "../data/mnli", 
                     device = "cuda:0", 
                     save_data = True,
                     data_label = "mnli",
                     tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")):
    mnli_dataset = load_dataset("multi_nli")
    mnli_pandas = mnli_dataset['train'].to_pandas()[["premise", "hypothesis", "label"]]
    mnli_pandas = pd.concat((mnli_pandas, mnli_dataset['validation_matched'].to_pandas()[["premise", "hypothesis", "label"]]), ignore_index=True)
    mnli_pandas = pd.concat((mnli_pandas, mnli_dataset['validation_mismatched'].to_pandas()[["premise", "hypothesis", "label"]]), ignore_index=True)
    mnli_pandas = mnli_pandas.sample(frac = 1, random_state=42)

    train_data, train_labels, test_data, test_labels = filter_data(mnli_pandas, 
                                                                   tokenizer,
                                                                   longest_sequence_allowed,
                                                                   make_binary,
                                                                   n_test_data,
                                                                   device)
    if save_data:
        save_data_to_file(train_data, 
                          train_labels, 
                          test_data, 
                          test_labels, 
                          data_path,
                          data_label)
    return train_data, train_labels, test_data, test_labels

def get_anli_dataset(model_name = "bert-base-uncased", 
                     longest_sequence_allowed = 64, 
                     n_test_data = 10000, 
                     make_binary = True, 
                     data_path = "../data/anli", 
                     device = "cuda:0", 
                     save_data = True,
                     data_label = "anli",
                     tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")):
    anli_dataset = load_dataset("anli")
    anli_pandas = anli_dataset['train_r3'].to_pandas()[["premise", "hypothesis", "label"]]
    anli_pandas = pd.concat((anli_pandas, anli_dataset['test_r3'].to_pandas()[["premise", "hypothesis", "label"]]), ignore_index=True)
    anli_pandas = pd.concat((anli_pandas, anli_dataset['dev_r3'].to_pandas()[["premise", "hypothesis", "label"]]), ignore_index=True)
    anli_pandas = anli_pandas.sample(frac = 1, random_state=42)
    train_data, train_labels, test_data, test_labels = filter_data(anli_pandas, 
                                                                   tokenizer,
                                                                   longest_sequence_allowed,
                                                                   make_binary,
                                                                   n_test_data,
                                                                   device)
    print(len(train_data), len(test_data))
    if save_data:
        save_data_to_file(train_data, 
                          train_labels, 
                          test_data, 
                          test_labels, 
                          data_path,
                          data_label)
    return train_data, train_labels, test_data, test_labels

def get_open_assistant_dataset(model_name = "gpt2",
                     longest_sequence_allowed = 512,
                     n_test_data = 10000,
                     data_path = "../data/open_assistant",
                     device = "cuda:0",
                     save_data = True,
                     data_label = "open_assistant",
                     tokenizer = AutoTokenizer.from_pretrained("gpt2")):

    open_assistant_dataset = load_dataset("OpenAssistant/oasst1")
    df_train = open_assistant_dataset['train'].to_pandas()
    df_test = open_assistant_dataset['validation'].to_pandas()

    train_convos = extract_convo_trees(df_train)
    test_convos = extract_convo_trees(df_test)

    train_convos_ids = torch.tensor(tokenizer.batch_encode_plus(train_convos, padding = "longest", max_length = longest_sequence_allowed, truncation = True)['input_ids']).to(device)
    test_convos_ids = torch.tensor(tokenizer.batch_encode_plus(test_convos, padding = "longest", max_length = longest_sequence_allowed, truncation = True)['input_ids']).to(device)


    if save_data:
        train_labels, test_labels = None, None
        save_data_to_file(train_convos_ids,
                          train_labels,
                          test_convos_ids,
                          test_labels,
                          data_path,
                          data_label)

    return train_convos_ids, train_convos_ids, test_convos_ids, test_convos_ids


def load_data(model_name, dataset_label, load_saved_data, data_path, longest_sequence_allowed, test_set_size, make_binary, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if dataset_label == "mnli":
        if str(load_saved_data) == "true":
            train_data, train_labels, test_data, test_labels = load_data_from_file(data_path, "mnli")
        else:
            train_data, train_labels, test_data, test_labels = get_mnli_dataset(model_name = model_name,
                                                                                longest_sequence_allowed = longest_sequence_allowed,
                                                                                n_test_data = test_set_size,
                                                                                make_binary = make_binary,
                                                                                data_path = data_path,
                                                                                device = device,
                                                                                save_data = True,
                                                                                tokenizer = tokenizer)
    elif dataset_label == "anli":
        if str(load_saved_data) == "true":
            train_data, train_labels, test_data, test_labels = load_data_from_file(data_path, "anli")
        else:
            train_data, train_labels, test_data, test_labels = get_anli_dataset(model_name = model_name,
                                                                                longest_sequence_allowed = longest_sequence_allowed,
                                                                                n_test_data = test_set_size,
                                                                                make_binary = make_binary,
                                                                                data_path = data_path,
                                                                                device = device,
                                                                                save_data = True,
                                                                                tokenizer = tokenizer)

    elif dataset_label == "open_assistant":
        if str(load_saved_data) == "true":
            train_data, train_labels, test_data, test_labels = load_data_from_file(data_path, "open_assistant")
        else:
            train_data, train_labels, test_data, test_labels = get_open_assistant_dataset(model_name = model_name,
                                                                                longest_sequence_allowed = longest_sequence_allowed,
                                                                                n_test_data = test_set_size,
                                                                                data_path = data_path,
                                                                                device = device,
                                                                                save_data = True,
                                                                                tokenizer = tokenizer)
    return train_data, train_labels, test_data, test_labels, tokenizer



def save_data_to_file(train_data, 
                      train_labels, 
                      test_data, 
                      test_labels, 
                      path,
                      dataset_label):
    data_save_relative_path = path + "/" + dataset_label
    print("Attempting to save data to relative path: \"" + data_save_relative_path + "_[stuff].pth" + "\"")
    torch.save(train_data, data_save_relative_path + "_training_data.pth")
    if not train_labels is None:
        torch.save(train_labels, data_save_relative_path + "_training_labels.pth")
    torch.save(test_data, data_save_relative_path + "_testing_data.pth")
    if not test_labels is None:
        torch.save(test_labels, data_save_relative_path + "_testing_labels.pth")

def load_data_from_file(path, dataset_label):
    data_load_relative_path = path + "/" + dataset_label
    print("Attempting to load data from relative path: \"" + data_load_relative_path + "_[stuff].pth" + "\"")
    train_data = torch.load(data_load_relative_path + "_training_data.pth")
    test_data = torch.load(data_load_relative_path + "_testing_data.pth")
    if not str.lower(dataset_label) in ['open_assistant']:
        train_labels = torch.load(data_load_relative_path + "_training_labels.pth")
        test_labels = torch.load(data_load_relative_path + "_testing_labels.pth")
    else:
        train_labels, test_labels = train_data, test_data
    return train_data, train_labels, test_data, test_labels

def tokenize_data(data, tokenizer, max_length = 128):
    text_data = []
    for i in range(len(data)):
        current_text = data.iloc[i]["premise"] + " [SEP] " + data.iloc[i]["hypothesis"]
        text_data.append(current_text)
    return torch.tensor(tokenizer.batch_encode_plus(text_data, padding = "longest", max_length = max_length, truncation = True)['input_ids'])

def filter_data(data, 
                tokenizer,
                longest_sequence_allowed = 64,
                make_binary = True,
                n_test_data = 10000,
                device = "cpu"):
    current_tokens = tokenize_data(data, tokenizer, max_length = longest_sequence_allowed + 10)
    inputs_too_long_flag = torch.sum(current_tokens > 0, dim = 1) <= longest_sequence_allowed
    inputs_short_enough_entries = [i for i in range(len(data)) if inputs_too_long_flag[i]]

    short_mnli_data = data.iloc[inputs_short_enough_entries]
    short_mnli_labels = torch.from_numpy(data.iloc[inputs_short_enough_entries]["label"].values)
    if make_binary:
        binary_labels_selector_flag = short_mnli_labels != 1
        inputs_only_entailment_or_contradiction_entries = [i for i in range(len(short_mnli_labels)) if binary_labels_selector_flag[i]]
        selected_labels = (short_mnli_labels[inputs_only_entailment_or_contradiction_entries] / 2).to(int)
        selected_tokens = tokenize_data(short_mnli_data.iloc[inputs_only_entailment_or_contradiction_entries], tokenizer)
    else:
        selected_labels = short_mnli_labels
        selected_tokens = tokenize_data(short_mnli_data, tokenizer)

    train_set_size = len(selected_tokens) - n_test_data
    random_indices_perm = torch.randperm(len(selected_tokens))
    train_set_selection_indices = random_indices_perm[:train_set_size]
    test_set_selection_indices = random_indices_perm[train_set_size:]

    train_data = selected_tokens[train_set_selection_indices].to(device)
    train_labels = selected_labels[train_set_selection_indices].to(device)
    test_data = selected_tokens[test_set_selection_indices].to(device)
    test_labels = selected_labels[test_set_selection_indices].to(device)
    return train_data, train_labels, test_data, test_labels
