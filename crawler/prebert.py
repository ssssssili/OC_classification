import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup


def preprocess_text(text):
    # Lowercase the text
    return text.lower()


def train_and_save_model(model_type, unfrozen_layers, text_data, num_epochs=3):
    # Preprocess the text
    text_data = preprocess_text(text_data)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(model_type)
    model = BertForSequenceClassification.from_pretrained(model_type).to(device)
    # Move the model to the GPU

    # Tokenize the input text and move tensors to GPU
    inputs = tokenizer(text_data, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    labels = torch.tensor([1]).unsqueeze(0).to(device)  # Replace 1 with the actual label you have for the text data

    # Freeze layers based on the 'unfrozen_layers' parameter
    for i, layer in enumerate(model.base_model.encoder.layer):
        for param in layer.parameters():
            param.requires_grad = False

    for layer_idx in unfrozen_layers:
        for param in model.base_model.encoder.layer[layer_idx].parameters():
            param.requires_grad = True

    # Create the optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=1e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=len(inputs["input_ids"]) * num_epochs)

    # Train the model
    for epoch in range(num_epochs):
        model.train()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        model.zero_grad()

    # Save the trained model
    model_save_path = f"{model_type}_layers{'_'.join(map(str, unfrozen_layers))}_model.pt"
    model.save_pretrained(model_save_path)

    return model_save_path


# Load text data from a text file
text_file_path = "isco88index.txt"
with open(text_file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

# Define the model types and layers to tune
model_types = ['bert-base-uncased', 'bert-base-multilingual-uncased']
layers_to_tune = [[-1], [-1, -2], ['all']]  # Examples of different layer combinations

# Train and save models with different parameters
for model_type in model_types:
    for unfrozen_layers in layers_to_tune:
        model_save_path = train_and_save_model(model_type, unfrozen_layers, text_data)
        print(f"Model saved at: {model_save_path}")
