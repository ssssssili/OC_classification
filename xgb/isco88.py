import pandas as pd
import numpy as np
import data_preprocess
import xgboost as xgb
import matplotlib as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

isco88_prep = pd.read_csv('../data/isco88_prep.csv')
isco88_data = data_preprocess.CombineFeature(isco88_prep, column=['occupation_en', 'task_en', 'employer_en', 'product_en'], withname=False)
isco88_data['label'] = isco88_data['isco88_cg_4']
isco88_data = isco88_data[['feature', 'label']]

embedding_model = data_preprocess.EmbeddingModel("roberta-base")
batch_size = 256
num_batches = len(isco88_data) // batch_size + 1
embeddings = []
labels = []

for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = start_idx + batch_size
    batch_texts = isco88_data['feature'][start_idx:end_idx]
    batch_embeddings = embedding_model.sentence_embedding(batch_texts)
    embeddings.append(batch_embeddings)
embeddings = np.concatenate(embeddings, axis=0)

le = LabelEncoder()
labels = le.fit_transform(isco88_data['label'])
x_train, x_test, x_val, y_train, y_test, y_val = data_preprocess.SplitDataset(embeddings, labels, 0.6, 0.3)

num_class = np.unique(labels)
all_parameters = {'objective': 'multi:softmax',
                    'num_class': num_class,
                    'gamma': 0.2,
                    'learning_rate': 0.05,
                    'n_estimators': 500,
                    'max_depth': 6,
                    'min_child_weight': 6,
                    'early_stopping_rounds': 10,
                    #'scale_pos_weight': 1,
                    'tree_method': 'gpu_hist',
                    'eval_metric': ['merror','mlogloss'],
                    'seed': 42}
xg = xgb.XGBClassifier(**all_parameters)
xg.fit(x_train,y_train,verbose=1,eval_set=[(x_train, y_train), (x_val, y_val)])

results = xg.evals_result()
epochs = len(results['validation_0']['mlogloss'])
x_axis = range(0, epochs)

# xgboost 'mlogloss' plot
fig, ax = plt.subplots(figsize=(9,5))
ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
ax.plot(x_axis, results['validation_1']['mlogloss'], label='Val')
ax.legend()
plt.ylabel('mlogloss')
plt.title('isco88 XGBoost mlogloss', fontdict={'size': 20})
plt.savefig('result/isco88 mlogloss.png')

# xgboost 'merror' plot
fig, ax = plt.subplots(figsize=(9,5))
ax.plot(x_axis, results['validation_0']['merror'], label='Train')
ax.plot(x_axis, results['validation_1']['merror'], label='Val')
ax.legend()
plt.ylabel('merror')
plt.title('isco88 XGBoost merror', fontdict={'size': 20})
plt.savefig('result/isco88 merror.png')

y_pred = xg.predict(x_test)

print('\n------------------ Evaluation Matrix -----------------\n')
print('\nAccuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))
print('Micro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='macro')))
print('Micro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='macro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='macro')))
print('Cohens Kappa: {:.2f}\n'.format(cohen_kappa_score(y_test, y_pred)))

print('\n--------------- Classification Report ---------------\n')
print(classification_report(y_test, y_pred))

print('\n------------------ Confusion Matrix -----------------\n')
np.savetxt('result/isco88 y_pred.txt', np.concatenate((y_pred[:,np.newaxis], y_test[:,np.newaxis]),axis=1))
print(confusion_matrix(y_test, y_pred))

print('---------------------- XGBoost ----------------------')
