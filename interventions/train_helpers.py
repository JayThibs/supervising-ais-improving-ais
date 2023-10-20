import torch
import transformers
from transformers import AutoModel, AutoModelForSequenceClassification, AutoModelForCausalLM
import copy
from connectivity_helpers import *
from eval_helpers import *
import tqdm
import random

def train_new_model(model,
                    train_data,
                    train_labels,
                    test_data,
                    test_labels,
                    training_generator = False,
                    existing_models = [],
                    optimizer = "adamw",
                    lr = 2e-3,
                    weight_decay = 0,
                    batch_size = 128,
                    epochs = 3,
                    warmup_frac = 0.1,
                    num_schedule_cycles = 1,
                    max_interpolation_model_loss = 1,
                    basin_exploration_loss_weight = 1,
                    device = "cuda:0",
                    test_each_update_step = False,
                    standard_deviation = 0.2,
                    pad_token_id = 0):
    all_train_losses = []
    all_test_losses = []
    all_train_accuracies = []
    all_test_accuracies = []

    n_train_data = len(train_data)
    steps_per_epoch = n_train_data // batch_size + (n_train_data % batch_size > 0)
    total_steps = steps_per_epoch * epochs

    if str.lower(optimizer) == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)
        optimizer_order = 1

    schedule = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
                                                                               num_warmup_steps = int(total_steps * warmup_frac),
                                                                               num_training_steps = total_steps,
                                                                               num_cycles = num_schedule_cycles)
    best_model_test_loss = 1000000
    best_model = AutoModelForCausalLM.from_config(model.config).to(device) if training_generator else \
                 AutoModelForSequenceClassification.from_config(model.config).to(device) 
    output_model = AutoModelForCausalLM.from_config(model.config).to(device) if training_generator else \
                   AutoModelForSequenceClassification.from_config(model.config).to(device)
    for epoch in range(epochs):
        n_basin_reg_loss_overflows = 0
        epoch_total_train_loss = 0
        epoch_total_basin_reg_loss = 0
        epoch_train_n_correct = 0
        epoch_train_n_basin_correct = 0
        epoch_n_preds = 0
        epoch_n_basin_preds = 0
        pbar = tqdm.tqdm(range(steps_per_epoch))

        randindex = list(range(n_train_data))
        random.shuffle(randindex)
        train_data = train_data[randindex]
        train_labels = train_labels[randindex]

        for step in pbar:
            batch_start = step * batch_size
            batch_end = min((step + 1) * batch_size, n_train_data)
            batch_data = train_data[batch_start: batch_end]
            batch_labels = train_labels[batch_start: batch_end]
            attention_mask = batch_data != pad_token_id
            batch_n_preds = torch.sum(attention_mask).item() if training_generator else len(batch_labels)
            epoch_n_preds += batch_n_preds

            #print(pad_token_id, attention_mask, batch_data)
            outputs = model(batch_data, labels = batch_labels, attention_mask = attention_mask)
            classification_loss = outputs.loss
            interpolation_model_classification_loss = torch.tensor([0])
            logits = outputs.logits
            argmax_dim = len(logits.size()) - 1
            if optimizer_order == 1:
                classification_loss.backward()
            elif optimizer_order == 2:
                classification_loss.backward(create_graph = True)
            
            for prior_model in existing_models:
                prior_model_weight = (torch.randn(1).item() * standard_deviation) + 0.5
                if prior_model_weight < 0 or prior_model_weight > 1:
                    prior_model_weight = 0.5
                linear_interpolation_sample_model = linear_model_mix(model, prior_model, prior_model_weight, output_model, device)
                prior_model_outputs = linear_interpolation_sample_model(batch_data, labels = batch_labels, attention_mask = attention_mask)
                interpolation_model_classification_loss = prior_model_outputs.loss

                if interpolation_model_classification_loss > max_interpolation_model_loss:
                    interpolation_model_classification_loss = interpolation_model_classification_loss * 0 + 1 # should be the same as just setting it equal to a constant, in terms of gradients
                    n_basin_reg_loss_overflows += 1
                    #print("Max interpolation regularizer loss encountered.")

                interpolation_model_classification_loss.backward(retain_graph = True)
                grads = list()
                for param in linear_interpolation_sample_model.parameters():
                    grads.append(param.grad.clone())

                for param, grad in zip(model.parameters(), grads):
                    param.grad += - basin_exploration_loss_weight * grad * 4*(prior_model_weight)*(1-prior_model_weight) / len(existing_models) # Strange prior model weight polynomial made to not punish model for being good.
                
                epoch_n_basin_preds += batch_n_preds

                basin_predictions = torch.argmax(prior_model_outputs.logits, dim=argmax_dim)
                basin_n_correct = torch.sum(basin_predictions == batch_labels).item()
                epoch_train_n_basin_correct += basin_n_correct
                epoch_total_basin_reg_loss += (interpolation_model_classification_loss * batch_n_preds).item()

            optimizer.step()
            schedule.step()
            optimizer.zero_grad()

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=argmax_dim)
            #print(logits.size(), batch_labels.size(), predictions.size())

            n_correct = torch.sum(predictions == batch_labels).item()
            #return n_correct, logits, predictions, batch_labels
            epoch_total_train_loss += (classification_loss * batch_n_preds).item()
            epoch_train_n_correct += n_correct
            pbar.set_description("CLS Loss: " + str(round(classification_loss.item(), 5)) + " -- Reg Loss: " + str(round(interpolation_model_classification_loss.item(), 5))) 

            #if test_each_update_step:
            #    step_test_loss, step_test_accuracy = eval_model(model, test_data, test_labels, batch_size = batch_size)
            #    if step_test_loss < best_model_test_loss:
            #        best_model = copy.deepcopy(model)
            #        best_model_test_loss = step_test_loss
            #        print("Updated best model:")
            #        print("New test loss    :", round(step_test_loss, 5))
            #        print("New test accuracy:", round(step_test_accuracy, 5))
        train_loss = epoch_total_train_loss / max(epoch_n_preds, 1)
        train_accuracy = epoch_train_n_correct / max(epoch_n_preds, 1)
        basin_reg_loss = epoch_total_basin_reg_loss / max(epoch_n_preds, 1)
        basin_reg_accuracy = epoch_train_n_basin_correct / max(epoch_n_basin_preds, 1)

        test_loss, test_accuracy = eval_model(model, test_data, test_labels, batch_size = batch_size, pad_token_id = pad_token_id)
        print("###########################")
        print("Epoch", epoch, ":")
        print("Train loss.       :", round(train_loss, 5))
        print("Train accuracy    :", round(train_accuracy, 5))
        print("Test loss.        :", round(test_loss, 5))
        print("Test accuracy     :", round(test_accuracy, 5))
        print("Basin reg loss.   :", round(basin_reg_loss, 5))
        print("Basin reg acc.    :", round(basin_reg_accuracy, 5))
        print("Reg loss overflows:", n_basin_reg_loss_overflows)

        all_train_losses.append(train_loss)
        all_test_losses.append(test_loss)
        all_train_accuracies.append(train_accuracy)
        all_test_accuracies.append(test_accuracy)

        if test_loss < best_model_test_loss:
            best_model = copy.deepcopy(model)
            best_model_test_loss = test_loss
    return model, best_model, best_model_test_loss, all_train_losses, all_train_accuracies, all_test_losses, all_test_accuracies
