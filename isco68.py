import pandas as pd
import numpy as np
import data_preprocess
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


lifew = pd.read_csv('data/(Dutch - ISCO-68) AMIGO_t - Copy.csv', encoding='latin-1')

isco68_short = lifew[lifew['obsseqnr'].astype(str).str.len()==4]

#isco68_prep = data_preprocess.PrepData(isco68_short, column=['bjobcoder', 'bwhsID', 'bjobnm', 'bjobdes',
 #      'bjobco', 'bjobcode', 'bjobcertain'], lan='dutch', lower=True, punc=True, stop_word=True, stemming=True)
#print(isco68_prep.head())
#isco68_prep.to_csv('isco68_prep.csv', index=False)

isco68_prep = pd.read_csv('isco68_prep.csv')

isco68_feature = data_preprocess.CombineFeature(isco68_prep, column=['bjobcoder', 'bwhsID', 'bjobnm', 'bjobdes',
                                                                 'bjobco', 'bjobcode', 'bjobcertain'], withname= False)

isco68_namefeature = data_preprocess.CombineFeature(isco68_prep, column=['bjobcoder', 'bwhsID', 'bjobnm', 'bjobdes',
                                                                 'bjobco', 'bjobcode', 'bjobcertain'], withname= True)

#data_preprocess.PlotData(isco68_prep, column='obsseqnr')

x = isco68_feature['feature']
y = isco68_feature['obsseqnr']

x_name = isco68_namefeature['feature']
y_name = isco68_namefeature['obsseqnr']

embedding_model = data_preprocess.EmbeddingModel("pdelobelle/robbert-v2-dutch-base")

texts_train1, texts_test1, labels_train1, labels_test1 = train_test_split(x, y, test_size=0.3, random_state=42)
texts_train2, texts_test2, labels_train2, labels_test2 = train_test_split(x_name, y_name, test_size=0.3, random_state=42)

# Define batch size for processing
batch_size = 128

# Calculate the number of batches
num_batches1 = len(texts_train1) // batch_size + 1
num_batches2 = len(texts_train2) // batch_size + 1

# Initialize empty arrays for storing embeddings and labels
embeddings1 = []
batch_labels1 = []

embeddings2 = []
batch_labels2 = []

# Process data in batches
for i in range(num_batches1):
    start_idx = i * batch_size
    end_idx = start_idx + batch_size

    batch_texts = texts_train1[start_idx:end_idx]
    batch_labels1.extend(labels_train1[start_idx:end_idx])

    batch_embeddings = embedding_model.sentence_embedding(batch_texts)
    embeddings1.append(batch_embeddings)

for i in range(num_batches2):
    start_idx = i * batch_size
    end_idx = start_idx + batch_size

    batch_texts = texts_train2[start_idx:end_idx]
    batch_labels2.extend(labels_train2[start_idx:end_idx])

    batch_embeddings = embedding_model.sentence_embedding(batch_texts)
    embeddings2.append(batch_embeddings)

# Concatenate embeddings and labels
embeddings1 = np.concatenate(embeddings1, axis=0)
labels_train1 = np.array(batch_labels1)

embeddings2 = np.concatenate(embeddings2, axis=0)
labels_train2 = np.array(batch_labels2)

# Convert the embeddings and labels to NumPy arrays
X_train1 = xgb.DMatrix(embeddings1, label=labels_train1)
X_train2 = xgb.DMatrix(embeddings2, label=labels_train2)

# Train the XGBoost model
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'error',
    'seed': 42
}

num_rounds = 100
model1 = xgb.train(params, X_train1, num_rounds)
model2 = xgb.train(params, X_train2, num_rounds)

# Perform inference on the test set
batch_size = 256

num_test_batches1 = len(texts_test1) // batch_size + 1
predictions1 = []

num_test_batches2 = len(texts_test2) // batch_size + 1
predictions2 = []

for i in range(num_test_batches1):
    start_idx = i * batch_size
    end_idx = start_idx + batch_size

    batch_texts = texts_test1[start_idx:end_idx]
    batch_embeddings = embedding_model.sentence_embedding(batch_texts)

    dtest = xgb.DMatrix(batch_embeddings)
    batch_predictions = model1.predict(dtest)

    predictions1.extend(batch_predictions)

for i in range(num_test_batches2):
    start_idx = i * batch_size
    end_idx = start_idx + batch_size

    batch_texts = texts_test2[start_idx:end_idx]
    batch_embeddings = embedding_model.sentence_embedding(batch_texts)

    dtest = xgb.DMatrix(batch_embeddings)
    batch_predictions = model2.predict(dtest)

    predictions2.extend(batch_predictions)

predictions1 = np.array(predictions1)
binary_predictions1 = np.round(predictions1)

predictions2 = np.array(predictions2)
binary_predictions2 = np.round(predictions2)

# Calculate accuracy
accuracy = accuracy_score(labels_test1, binary_predictions1)
accuracy = accuracy_score(labels_test2, binary_predictions2)
print("Accuracy:", accuracy)



