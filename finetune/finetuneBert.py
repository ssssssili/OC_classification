import os
import torch
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ExponentialLR


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

def fine_tune_bert(feature, label, model_path, unfreeze_layers, batch_size, num_epochs, max_length, num_labels, name):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """
    print(torch.cuda.current_device())
    print(device)
    exit()
    """

    le = LabelEncoder()
    labels = le.fit_transform(label).tolist()
    texts = feature.tolist()

    train_texts, temp_texts, train_labels, temp_labels = train_test_split(texts, labels, test_size=0.4, random_state=42)
    val_texts, test_texts, val_labels, test_labels = train_test_split(temp_texts, temp_labels, test_size=0.75,
                                                                      random_state=42)

    # Initialize the BERT model configuration
    config = BertConfig.from_pretrained(model_path, num_labels=num_labels)

    # Initialize the BERT model for sequence classification
    model = BertForSequenceClassification(config).to(device)
    tokenizer = BertTokenizer.from_pretrained(model_path)

    # Create datasets and data loaders
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length=max_length)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length=max_length)
    test_dataset = TextClassificationDataset(test_texts, test_labels, tokenizer, max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Freeze layers before the specified layers to be unfrozen
    for name, param in model.named_parameters():
        param.requires_grad = False
        for item in unfreeze_layers:
            if item in name:
                param.requires_grad = True
                break

    # Define optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
    loss_fn = torch.nn.CrossEntropyLoss()
    #scheduler = ExponentialLR(optimizer, gamma=0.9)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    # Early stopping
    best_val_loss = float('inf')
    patience = 5

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Calculate the loss
            loss = loss_fn(logits, labels)

            # Backpropagation
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        # Calculate average training loss for the epoch
        avg_train_loss = total_train_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{num_epochs} - Average training loss: {avg_train_loss:.4f}')
        #scheduler.step(total_train_loss)

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
            torch.save(model.state_dict(), f"{name}_best_model.pt")  # Save the best model so far
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping!")
                break

    # Load the best model and evaluate it on the test set
    model.load_state_dict(torch.load(f"{name}_best_model.pt"))
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
            logits = outputs.logits

            predictions = torch.argmax(logits, dim=1)
            test_predictions.extend(predictions.cpu().tolist())
            test_true_labels.extend(labels.cpu().tolist())

    # Synchronize the GPU after the evaluation
    torch.cuda.synchronize()

    # Calculate evaluation metrics
    accuracy = accuracy_score(test_true_labels, test_predictions)
    precision = precision_score(test_true_labels, test_predictions, average='macro')
    recall = recall_score(test_true_labels, test_predictions, average='macro')
    f1 = f1_score(test_true_labels, test_predictions, average='macro')
    cohen_kappa = cohen_kappa_score(test_true_labels, test_predictions)

    # Save the fine-tuned model
    model.save_pretrained(f"{name}_fine_tuned_model")

    evaluation_results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'cohen_kappa': cohen_kappa,
        'test_true_labels': le.inverse_transform(test_true_labels),
        'test_predictions': le.inverse_transform(test_predictions)
    }

    return evaluation_results


def train_and_evaluate_series_model(feature, label, model_type, layer_configs, batch_size, num_epochs, max_length,
                                    num_labels, name, result_filename, test_labels_filename, test_predictions_filename):
    best_evaluation_results = None
    best_model_name = None
    best_model_config_num = None
    best_val_accuracy = 0.0

    results = {}

    for config_num, unfreeze_layers in enumerate(layer_configs, 1):
        print(f"Training configuration {config_num} - Unfreezing layers: {unfreeze_layers}")
        model_name = f"model_config_{config_num}"

        # Fine-tune and evaluate the model
        evaluation_results = fine_tune_bert(feature, label, model_path=model_type, unfreeze_layers=unfreeze_layers,
                                            batch_size=batch_size, num_epochs=num_epochs, max_length=max_length,
                                            num_labels=num_labels, name=f"{name}_{config_num}")

        results[model_name] = evaluation_results
        print("Test Accuracy:", evaluation_results['accuracy'])
        print("Test Precision:", evaluation_results['precision'])
        print("Test Recall:", evaluation_results['recall'])
        print("Test F1 Score:", evaluation_results['f1_score'])
        print("Test Cohen's Kappa:", evaluation_results['cohen_kappa'])
        print("-----------------------------")

        # Check if the current model performs better than the previous best model
        if evaluation_results['accuracy'] > best_val_accuracy:
            best_val_accuracy = evaluation_results['accuracy']
            best_evaluation_results = evaluation_results
            best_model_name = model_name
            best_model_config_num = config_num

    # Save the best-performing model's evaluation results to a file
    print(f"Best Model Configuration: {best_model_config_num}\n")
    print(f"Model Name: {best_model_name}\n")
    print(f"Validation Accuracy: {best_evaluation_results['accuracy']}\n")
    print(f"Validation Precision: {best_evaluation_results['precision']}\n")
    print(f"Validation Recall: {best_evaluation_results['recall']}\n")
    print(f"Validation F1 Score: {best_evaluation_results['f1_score']}\n")
    print(f"Validation Cohen's Kappa: {best_evaluation_results['cohen_kappa']}\n")

    # Save the test labels and predicted labels of the best-performing model
    test_true_labels = best_evaluation_results['test_true_labels']
    test_predictions = best_evaluation_results['test_predictions']

    with open(test_labels_filename, 'w') as file:
        file.write('\n'.join(str(label) for label in test_true_labels))

    with open(test_predictions_filename, 'w') as file:
        file.write('\n'.join(str(label) for label in test_predictions))

    return results
