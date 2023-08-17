import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

def preprocess_text(text, max_length=512):
    # Lowercase the text
    text = text.lower()
    # Split text into chunks of maximum length
    text_chunks = [text[i:i + max_length] for i in range(0, len(text), max_length)]
    return text_chunks

def train_and_save_model(model_type, unfrozen_layers, text_data, num_epochs=10):
    # Preprocess the text
    text_chunks = preprocess_text(text_data)

    # Load the tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(model_type)
    model = BertForSequenceClassification.from_pretrained(model_type)
    model.to("cuda")  # Move the model to the GPU

    # Tokenize the input text chunks and move tensors to GPU
    tokenized_chunks = [tokenizer(chunk, return_tensors="pt", truncation=True, padding=True, max_length=512) for chunk
                        in text_chunks]
    tokenized_chunks = [{k: v.to("cuda") for k, v in inputs.items()} for inputs in tokenized_chunks]

    """
    # Determine which layers to unfreeze
    if unfrozen_layers == 'all':
        unfrozen_layers = list(range(len(model.base_model.encoder.layer)))
    else:
        # Convert layer indices to a list if not already a list
        unfrozen_layers = [unfrozen_layers] if isinstance(unfrozen_layers, int) else unfrozen_layers
    """

    # Freeze/unfreeze layers
    for i, layer in enumerate(model.base_model.encoder.layer):
        for param in layer.parameters():
            param.requires_grad = False

    for layer_idx in unfrozen_layers:
        for param in model.base_model.encoder.layer[layer_idx].parameters():
            param.requires_grad = True

    # Create the optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=1e-5)
    #scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                               # num_training_steps=len(text_chunks) * num_epochs)

    # Train the model
    for epoch in range(num_epochs):
        model.train()
        for inputs in tokenized_chunks:
            labels = torch.tensor([1]).unsqueeze(0).to(
                "cuda")  # Replace 1 with the actual label you have for the text data

            # Training step
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            #scheduler.step()
            model.zero_grad()

    # Save the trained model
    model_save_path = f"{index}_{model_type}_layers{'_'.join(map(str, unfrozen_layers))}_model.pt"
    model.save_pretrained(model_save_path)

    return model_save_path

# Load text data from text file
text_file_path = "pcsindex.txt"
index = 'pcs'
with open(text_file_path, "r", encoding="utf-8") as file:
    text_data = file.readlines()
    print(text_data)

# Define the model types and layers to tune
model_type = 'bert-base-multilingual-uncased'
layers_to_tune = [[11], [10, 11]]  # Different layer combinations

# Train and save models with different parameters
for unfrozen_layers in layers_to_tune:
    model_save_path = train_and_save_model(index, model_type, unfrozen_layers, text_data, num_epochs=20)
    print(f"Model saved at: {model_save_path}")