import pandas as pd
import numpy as np
import data_preprocess
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.utils.class_weight import compute_sample_weight
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"


naf_prep = pd.read_csv('naf_prep.csv')

le = preprocessing.LabelEncoder()
naf_prep['label'] = le.fit_transform(naf_prep['code_naf'])

naf_feature = data_preprocess.CombineFeature(naf_prep, column=['numep', 'profession_txt',
                                                                  'secteur_txt'], withname= True)

naf_data = naf_feature[['feature', 'label']]

embedding_model = data_preprocess.EmbeddingModel("roberta-base-wechsel-french")
batch_size = 256
num_batches = len(naf_data) // batch_size + 1
embeddings = []
labels = []

for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = start_idx + batch_size

    batch_texts = naf_data['feature'][start_idx:end_idx]
    labels.extend(naf_data['label'][start_idx:end_idx])

    batch_embeddings = embedding_model.sentence_embedding(batch_texts)
    embeddings.append(batch_embeddings)

embeddings = np.concatenate(embeddings, axis=0)
labels = np.array(labels)


all_parameters = {'objective': 'multi:softmax',
                    'num_class': len(np.unique(naf_data['label'])),
                    'gamma': 0,
                    'learning_rate': 0.1,
                    'n_estimators': 500,
                    'max_depth': 5,
                    'early_stopping_rounds': 10,
                    'scale_pos_weight': 5,
                    'tree_method': 'gpu_hist',
                    'eval_metric': ['merror','mlogloss'],
                    'seed': 42}

x_train, x_test, x_val, y_train, y_test, y_val = data_preprocess.splitDataset(embeddings, labels, 0.6, 0.3)

xgbtree = xgb.XGBClassifier(**all_parameters)

xgbtree.fit(x_train,
            y_train,
            verbose=0, # set to 1 to see xgb training round intermediate results
            eval_set=[(x_train, y_train), (x_val, y_val)])

results = xgbtree.evals_result()
epochs = len(results['validation_0']['mlogloss'])
x_axis = range(0, epochs)

# xgboost 'mlogloss' plot
fig, ax = plt.subplots(figsize=(9,5))
ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
ax.plot(x_axis, results['validation_1']['mlogloss'], label='Val')
ax.legend()
plt.ylabel('mlogloss')
plt.title('GridSearchCV XGBoost mlogloss')
plt.show()

# xgboost 'merror' plot
fig, ax = plt.subplots(figsize=(9,5))
ax.plot(x_axis, results['validation_0']['merror'], label='Train')
ax.plot(x_axis, results['validation_1']['merror'], label='Val')
ax.legend()
plt.ylabel('merror')
plt.title('GridSearchCV XGBoost merror')
plt.show()

y_pred = xgbtree.predict(x_test)
print(y_pred)
np.savetxt('y_pred.txt', y_pred)

print('\n------------------ Confusion Matrix -----------------\n')

print('\nAccuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))
print('Micro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='macro')))
print('Micro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='macro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='macro')))
print('Cohens Kappa: {:.2f}\n'.format(cohen_kappa_score(y_test, y_pred)))

print('\n--------------- Classification Report ---------------\n')
print(classification_report(y_test, y_pred))
print('---------------------- XGBoost ----------------------')


isco88_index = data_preprocess.DataSplit(labels, thershold=5, num=3)

x1 = embeddings[[isco88_index[0]]]
y1 = labels[[isco88_index[0]]]

x2 = embeddings[[isco88_index[1]]]
y2 = labels[[isco88_index[1]]]

x3 = embeddings[[isco88_index[2]]]
y3 = labels[[isco88_index[2]]]

x_train1, x_test1, x_val1, y_train1, y_test1, y_val1 = data_preprocess.splitDataset(x1, y1, 0.6, 0.3)
x_train2, x_test2, x_val2, y_train2, y_test2, y_val2 = data_preprocess.splitDataset(x2, y2, 0.6, 0.3)
x_train3, x_test3, x_val3, y_train3, y_test3, y_val3 = data_preprocess.splitDataset(x3, y3, 0.6, 0.3)

x_test_e = np.concatenate((x_test1,x_test2,x_test3),axis=0)
y_test_e = np.concatenate((y_test1,y_test2,y_test3),axis=0)

model1 = xgb.XGBClassifier(**all_parameters)
model2 = xgb.XGBClassifier(**all_parameters)
model3 = xgb.XGBClassifier(**all_parameters)

model1.fit(x_train1,
            y_train1,
            verbose=0, # set to 1 to see xgb training round intermediate results
            eval_set=[(x_train1, y_train1), (x_val1, y_val1)])

model2.fit(x_train2,
            y_train2,
            verbose=0, # set to 1 to see xgb training round intermediate results
            eval_set=[(x_train2, y_train2), (x_val2, y_val2)])

model3.fit(x_train3,
            y_train3,
            verbose=0, # set to 1 to see xgb training round intermediate results
            eval_set=[(x_train3, y_train3), (x_val3, y_val3)])


# Load and initialize the three XGBoost models
model_path1 = 'model1.xgb'  # Path to the first XGBoost model
model_path2 = 'model2.xgb'  # Path to the second XGBoost model
model_path3 = 'model3.xgb'  # Path to the third XGBoost model

# Create an ensemble of the XGBoost models
y_pred_e = data_preprocess.ensemble_predict([model1, model2, model3], x_test_e)

print(y_pred_e)
np.savetxt('y_pred_e.txt', y_pred_e)

print('\n------------------ Confusion Matrix -----------------\n')

print('\nAccuracy: {:.2f}'.format(accuracy_score(y_test_e, y_pred_e)))
print('Micro Precision: {:.2f}'.format(precision_score(y_test_e, y_pred_e, average='macro')))
print('Micro Recall: {:.2f}'.format(recall_score(y_test_e, y_pred_e, average='macro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test_e, y_pred_e, average='macro')))
print('Cohens Kappa: {:.2f}\n'.format(cohen_kappa_score(y_test_e, y_pred_e)))

print('\n--------------- Classification Report ---------------\n')
print(classification_report(y_test_e, y_pred_e))
print('---------------------- EnsembleXGBoost ----------------------')








"""
model1 = xgb.XGBClassifier(objective='multi:softmax',
                            num_class=len(np.unique(y)),
                            gamma=0,
                            learning_rate=0.1,
                            n_estimators=500,
                            #missing=0,
                            max_depth=5,
                            #reg_lambda=1,
                            early_stopping_rounds=10,
                            scale_pos_weight= 5,
                            tree_method= 'gpu_hist',
                            eval_metric=['merror','mlogloss'],
                            seed=42

num_rounds = 100
model1 = xgb.train(params, X_train, num_rounds)

num_test_batches = len(texts_test) // batch_size + 1
predictions = []

for i in range(num_test_batches):
    batch_predictions = model1.predict(dtest)

    predictions.extend(batch_predictions)

predictions = np.argmax(predictions, axis=1)

accuracy1 = accuracy_score(labels_test1, predictions1)
f1score1 = f1_score(labels_test1, predictions1)
cohen1 = cohen_kappa_score(labels_test1, predictions1)

print("first: Accuracy, f1, cohen:", accuracy1, f1score1, cohen1)

isco88_index = data_preprocess.DataSplit(isco88_prep, column='label', thershold=5, num=3)

isco88_1 = isco88_prep.loc[isco88_prep['label'].isin(isco88_index[0])]
isco88_2 = isco88_prep.loc[isco88_prep['label'].isin(isco88_index[1])]
isco88_3 = isco88_prep.loc[isco88_prep['label'].isin(isco88_index[2])]
"""


"""
num_batches = len(x_train) // batch_size + 1
embeddings = []
labels = []

for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = start_idx + batch_size

    batch_texts = x_train[start_idx:end_idx]
    labels.extend(y_train[start_idx:end_idx])

    batch_embeddings = embedding_model.sentence_embedding(batch_texts)
    embeddings.append(batch_embeddings)

embeddings = np.concatenate(embeddings, axis=0)
labels = np.array(labels)

num_test_batches = len(x_test) // batch_size + 1
test_embedding = []
test_labels = []

for i in range(num_test_batches):
    start_idx = i * batch_size
    end_idx = start_idx + batch_size

    batch_texts = x_test[start_idx:end_idx]
    test_labels.extend(y_test[start_idx:end_idx])

    batch_embeddings = embedding_model.sentence_embedding(batch_texts)
    test_embedding.append(batch_embeddings)

test_embedding = np.concatenate(test_embedding, axis=0)
test_labels = np.array(test_labels)

num_val_batches = len(x_val) // batch_size + 1
val_embedding = []
val_labels = []

for i in range(num_val_batches):
    start_idx = i * batch_size
    end_idx = start_idx + batch_size

    batch_texts = x_val[start_idx:end_idx]
    val_labels.extend(y_val[start_idx:end_idx])

    batch_embeddings = embedding_model.sentence_embedding(batch_texts)
    val_embedding.append(batch_embeddings)

val_embedding = np.concatenate(val_embedding, axis=0)
val_labels = np.array(val_labels)

#x_train, y_train = data_preprocess.aggdata(isco68_feature, embeddings1, labels1)
#x_val, y_val = data_preprocess.aggdata(isco68_feature, val_embedding1, val_labels1)
"""

"""
objective='multi:softmax',
                            num_class=len(np.unique(isco88_data['label'])),
                            gamma=0,
                            learning_rate=0.1,
                            n_estimators=500,
                            #missing=0,
                            max_depth=5,
                            #reg_lambda=1,
                            early_stopping_rounds=10,
                            scale_pos_weight= 5,
                            tree_method= 'gpu_hist',
                            eval_metric=['merror','mlogloss'],
                            seed=42
"""