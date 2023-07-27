import pandas as pd
import numpy as np
import data_preprocess
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

isco68_prep = pd.read_csv('../data/isco68_prep.csv')
isco68_data = data_preprocess.CombineFeature(isco68_prep, column=['bjobnm'], withname= False)
isco68_data['label'] = isco68_data['bjobcode']
isco68_data = isco68_data[['feature', 'label']]

embedding_model = data_preprocess.EmbeddingModelB("bert-base-multilingual-uncased")
batch_size = 256
num_batches = len(isco68_data) // batch_size + 1
embeddings = []
labels = []

for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = start_idx + batch_size
    batch_texts = isco68_data['feature'][start_idx:end_idx]
    batch_embeddings = embedding_model.sentence_embedding(batch_texts)
    embeddings.append(batch_embeddings)
embeddings = np.concatenate(embeddings, axis=0)

le = LabelEncoder()
labels = le.fit_transform(isco68_data['label'])
x_train, x_test, x_val, y_train, y_test, y_val = data_preprocess.SplitDataset(embeddings, labels, 0.6, 0.3)

num_class = np.unique(labels)
all_parameters = {'objective': 'multi:softmax',
                    'num_class': num_class,
                    'gamma': 0.1,
                    'learning_rate': 0.05,
                    'n_estimators': 300,
                    'max_depth': 10,
                    'min_child_weight': 6,
                    #'alpha': 1,
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
plt.title('isco68 XGBoost mlogloss', fontdict={'size': 20})
plt.savefig('result/isco68 mlogloss.png')

# xgboost 'merror' plot
fig, ax = plt.subplots(figsize=(9,5))
ax.plot(x_axis, results['validation_0']['merror'], label='Train')
ax.plot(x_axis, results['validation_1']['merror'], label='Val')
ax.legend()
plt.ylabel('merror')
plt.title('isco68 XGBoost merror', fontdict={'size': 20})
plt.savefig('result/isco68 merror.png')

y_pred = xg.predict(x_test)

print('\n------------------ Evaluation Matrix -----------------\n')
print('\nAccuracy: {:.4f}'.format(accuracy_score(y_test, y_pred)))
print('Micro Precision: {:.4f}'.format(precision_score(y_test, y_pred, average='macro')))
print('Micro Recall: {:.4f}'.format(recall_score(y_test, y_pred, average='macro')))
print('Micro F1-score: {:.4f}\n'.format(f1_score(y_test, y_pred, average='macro')))
print('Cohens Kappa: {:.4f}\n'.format(cohen_kappa_score(y_test, y_pred)))

print('\n--------------- Classification Report ---------------\n')
print(classification_report(le.inverse_transform(y_test), le.inverse_transform(y_pred)))

np.savetxt('result/isco68 y_pred.txt', np.concatenate((le.inverse_transform(np.array(y_test))[:,np.newaxis],
                                                       le.inverse_transform(np.array(y_pred))[:,np.newaxis]),axis=1))
print('---------------------- XGBoost ----------------------')