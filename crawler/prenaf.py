from mlmtrain import train_bert_unsupervised

model_name = 'bert-base-multilingual-uncased'

unfreeze_layers = [
    ['predictions', 'layer.11'],
    ['predictions', 'layer.11', 'layer.10']
]

naf_index = 'naf'

with open("nafindex.txt", "r", encoding="utf-8") as file:
    naf_texts = file.readlines()
    naf_texts = str(naf_texts)

for layers in unfreeze_layers:
    train_bert_unsupervised(naf_index, model_name, naf_texts, layers)
