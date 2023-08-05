import torch
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
import os

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def fine_tune_bert(train_texts, train_labels, val_texts, val_labels, test_texts, test_labels,
                   model_path, unfreeze_layers, batch_size, num_epochs, max_length, num_labels):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    model = BertForSequenceClassification.from_pretrained(model_path, num_labels)
    tokenizer = BertTokenizer.from_pretrained(model_path, num_labels)

    # Create datasets and data loaders
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length=max_length)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length=max_length)
    test_dataset = TextClassificationDataset(test_texts, test_labels, tokenizer, max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Freeze layers before the specified layers to be unfrozen
    for param in model.base_model.parameters():
        param.requires_grad = False

    for layer in unfreeze_layers:
        for param in model.base_model.encoder.layer[layer].parameters():
            param.requires_grad = True

    # Define optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Early stopping
    best_val_loss = float('inf')
    patience = 5

    # Fine-tuning loop
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            model.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

        average_train_loss = total_train_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{num_epochs} - Average training loss: {average_train_loss:.4f}')

        # Validation
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_val_loss += loss.item()

        average_val_loss = total_val_loss / len(val_loader)
        print(f'Epoch {epoch + 1}/{num_epochs} - Average validation loss: {average_val_loss:.4f}')

        # Early stopping check
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pt")  # Save the best model so far
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping!")
                break

    # Load the best model and evaluate it on the test set
    model.load_state_dict(torch.load("best_model.pt"))
    model.eval()

    # Synchronize the GPU before the evaluation
    torch.cuda.synchronize()

    # Evaluation on the test set
    print("Evaluating on the test set...")
    test_predictions, test_true_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            predictions = torch.argmax(logits, dim=1)
            test_predictions.extend(predictions.cpu().tolist())
            test_true_labels.extend(labels.cpu().tolist())

    # Synchronize the GPU after the evaluation
    torch.cuda.synchronize()

    # Calculate evaluation metrics
    accuracy = accuracy_score(test_true_labels, test_predictions)
    precision = precision_score(test_true_labels, test_predictions)
    recall = recall_score(test_true_labels, test_predictions)
    f1 = f1_score(test_true_labels, test_predictions)
    cohen_kappa = cohen_kappa_score(test_true_labels, test_predictions)

    # Save the fine-tuned model
    model.save_pretrained("fine_tuned_model")

    evaluation_results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'cohen_kappa': cohen_kappa,
        'test_true_labels': test_true_labels,
        'test_predictions': test_predictions
    }

    return evaluation_results


def train_and_evaluate_series_model(train_texts, train_labels, val_texts, val_labels, test_texts, test_labels,
                                    model_type, layer_configs, batch_size, num_epochs, max_length, num_labels,
                                    result_filename, test_labels_filename, test_predictions_filename):
    best_evaluation_results = None
    best_model_name = None
    best_model_config_num = None
    best_val_accuracy = 0.0

    results = {}

    for config_num, unfreeze_layers in enumerate(layer_configs, 1):
        print(f"Training configuration {config_num} - Unfreezing layers: {unfreeze_layers}")
        model_name = f"model_config_{config_num}"

        # Fine-tune and evaluate the model
        evaluation_results = fine_tune_bert(train_texts, train_labels, val_texts, val_labels, test_texts, test_labels,
                                            model_path=model_type, unfreeze_layers=unfreeze_layers, batch_size=batch_size,
                                            num_epochs=num_epochs, max_length=max_length, num_labels=num_labels)

        results[model_name] = evaluation_results

        # Check if the current model performs better than the previous best model
        if evaluation_results['accuracy'] > best_val_accuracy:
            best_val_accuracy = evaluation_results['accuracy']
            best_evaluation_results = evaluation_results
            best_model_name = model_name
            best_model_config_num = config_num

    # Save the best-performing model's evaluation results to a file
    with open(result_filename, 'w') as file:
        file.write(f"Best Model Configuration: {best_model_config_num}\n")
        file.write(f"Model Name: {best_model_name}\n")
        file.write(f"Validation Accuracy: {best_evaluation_results['accuracy']}\n")
        file.write(f"Validation Precision: {best_evaluation_results['precision']}\n")
        file.write(f"Validation Recall: {best_evaluation_results['recall']}\n")
        file.write(f"Validation F1 Score: {best_evaluation_results['f1_score']}\n")
        file.write(f"Validation Cohen's Kappa: {best_evaluation_results['cohen_kappa']}\n")

    # Save the test labels and predicted labels of the best-performing model
    test_true_labels = best_evaluation_results['test_true_labels']
    test_predictions = best_evaluation_results['test_predictions']

    with open(test_labels_filename, 'w') as file:
        file.write('\n'.join(str(label) for label in test_true_labels))

    with open(test_predictions_filename, 'w') as file:
        file.write('\n'.join(str(label) for label in test_predictions))

    return results
