import torch
from transformers import BertTokenizer, BertForMaskedLM
from torch.utils.data import DataLoader, Dataset

# Step 1: Load your text data from the .txt file (if it's not already loaded)
with open('isco88index.txt', 'r', encoding='utf-8') as file:
    text_data = file.read()
    text_data = text_data.lower()

# Step 2: Tokenize and preprocess your text data
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
inputs = tokenizer(text_data, return_tensors='pt', padding=True, truncation=True)

# Step 3: Prepare the dataset and data loader
class CustomDataset(Dataset):
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx]
        }

# Create DataLoader from the custom dataset
dataset = CustomDataset(inputs['input_ids'], inputs['attention_mask'])
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Define the device (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# List of models with different configurations
models = [
    {
        'model_name': 'bert-base-multilingual-uncased',  # Base BERT
        'unfrozen_layers': [-2, -1],  # Unfreeze the last two layers
        'learning_rate': 2e-5
    },
    {
        'model_name': 'bert-base-multilingual-uncased',  # Multilingual BERT
        'unfrozen_layers': [-3, -2, -1],  # Unfreeze the last three layers
        'learning_rate': 2e-5
    },
    {
        'model_name': 'bert-base-multilingual-uncased',  # Base BERT
        'unfrozen_layers': 'all',  # Unfreeze all layers
        'learning_rate': 2e-5
    },
    {
        'model_name': 'bert-base-multilingual-uncased',  # Base BERT
        'unfrozen_layers': 'all',  # Unfreeze all layers
        'learning_rate': 1e-5
    }
]

for model_config in models:
    # Step 4: Load the pre-trained model and prepare for fine-tuning
    model_name = model_config['model_name']
    model = BertForMaskedLM.from_pretrained(model_name)
    model.train()
    model.to(device)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze specific layers for fine-tuning
    unfrozen_layers = model_config['unfrozen_layers']
    for layer_index in unfrozen_layers:
        for param in model.bert.encoder.layer[layer_index].parameters():
            param.requires_grad = True

    # Define the optimizer with different learning rate for unfrozen layers
    learning_rate = model_config['learning_rate']
    optimizer = torch.optim.AdamW([
        {'params': model.bert.embeddings.parameters(), 'lr': learning_rate},
        {'params': model.bert.pooler.parameters(), 'lr': learning_rate},
        {'params': model.bert.encoder.layer.parameters(), 'lr': learning_rate},
    ])

    # Step 5: Train the model with different layers unfrozen
    epochs = 5
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(dataloader)
        print(f"{model_name} - Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

    # Step 6: Save the fine-tuned model
    model_save_path = f'{model_name}_unfrozen_layers_{unfrozen_layers}_lr_{learning_rate}.pt'
    model.save_pretrained(model_save_path)
    print(f"Model saved to {model_save_path}")
