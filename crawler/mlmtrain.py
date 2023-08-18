from transformers import BertForMaskedLM, BertTokenizer, AdamW
import torch


# Chunking function
def process_text_chunks(text, chunk_size=512, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def train_bert_unsupervised(index, model_name, texts, unfreeze_layers, num_epochs=10, learning_rate=2e-5):
    # Load pre-trained BERT model and tokenizer for MLM
    model = BertForMaskedLM.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Identify which layers to update
    for name, param in model.named_parameters():
        param.requires_grad = False
        for item in unfreeze_layers:
            if item in name:
                param.requires_grad = True
                break

    # Set up optimizer for selected layers
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    # Training loop
    model.train()

    # Process and tokenize the long text
    text_chunks = process_text_chunks(texts)
    encoded_chunks = [tokenizer.encode_plus(chunk, add_special_tokens=True, padding='max_length', truncation=True,
                                          max_length=512, return_tensors='pt') for chunk in text_chunks]
    """
    input_ids = encoded_chunk['input_ids']
    tokenized_chunks = [tokenizer(chunk, return_tensors="pt", truncation=True, padding=True, max_length=512) for chunk
                        in text_chunks]
    tokenized_chunks = [{k: v.to("cuda") for k, v in inputs.items()} for inputs in tokenized_chunks]
    """

    for epoch in range(num_epochs):
        total_loss = 0.0

        for input_ids in encoded_chunks['input_ids']:

            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids=input_ids)
            loss = torch.nn.functional.cross_entropy(outputs.logits.view(-1, tokenizer.vocab_size), input_ids.view(-1))

            # Backpropagation
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss
        print(f"Average Loss: {average_loss:.4f}")
        if (epoch+1) / 3 == 0:
            path = f'{index}_epoch{epoch}_mlm.pt'
            model.save_pretrained(path)
            print(f'the model of epoch {epoch} is saved at {path}')

    model_path = f'{index}_epoch{num_epochs}_mlm.pt'
    model.save_pretrained(model_path)
    print(f'the model of epoch {num_epochs} is saved at {model_path}')