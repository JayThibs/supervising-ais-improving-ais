import torch
import transformers

def eval_model(model, data, labels, batch_size = 128, pad_token_id = 0):
    model = model.eval()
    n_data = len(data)
    n_preds = 0
    steps = n_data // batch_size + (n_data % batch_size > 0)
    total_loss = 0
    total_n_correct = 0
    with torch.no_grad():
        for step in range(steps):
            batch_start = step * batch_size
            batch_end = min((step + 1) * batch_size, n_data)
            batch_data = data[batch_start: batch_end]
            attention_mask = batch_data != pad_token_id
            batch_labels = labels[batch_start: batch_end]

            #print(batch_data, batch_labels, attention_mask, model)
            outputs = model(input_ids = batch_data, labels = batch_labels, attention_mask = attention_mask)
            loss = outputs.loss
            logits = outputs.logits

            batch_preds = torch.sum(attention_mask).item()
            n_preds += batch_preds

            predictions = torch.argmax(logits, dim=len(logits.size()) - 1)
            n_correct = torch.sum(predictions == batch_labels).item()
            total_loss += loss.item() * batch_preds
            total_n_correct += n_correct
    model = model.train()
    return total_loss / n_preds, total_n_correct / n_preds
