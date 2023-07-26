import pandas as pd
import numpy as np
import data_preprocess
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

print('\n------------------ isco68 xgb -----------------\n')
isco68_prep = pd.read_csv('../data/isco68_prep.csv')
isco68_data = data_preprocess.CombineFeature(isco68_prep, column=['bjobnm','bjobdes','bjobco'], withname= False)
isco68_data['label'] = isco68_data['bjobcode']
isco68_data = isco68_data[['feature', 'label']]

embedding_model = data_preprocess.EmbeddingModel("pdelobelle/robbert-v2-dutch-base")
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

le1 = LabelEncoder()
labels = le1.fit_transform(isco68_data['label'])
x_train, x_test, x_val, y_train, y_test, y_val = data_preprocess.SplitDataset(embeddings, labels, 0.6, 0.3)

num_class = np.unique(labels)
times = range(3)
max_depth = []
min_child = []
gamma = []
learning_rate = []
score = []

for learn in times:
  for gam in times:
    for min in times:
      for max in times:
        all_parameters = {'objective': 'multi:softmax',
                    'num_class': num_class,
                    'gamma': 0.1*gam,
                    'learning_rate': 0.05*(learn+1),
                    'n_estimators': 500,
                    'max_depth': max+4,
                    'min_child_weight': min+4,
                    'early_stopping_rounds': 10,
                    'scale_pos_weight': 1,
                    'tree_method': 'gpu_hist',
                    'eval_metric': ['merror','mlogloss'],
                    'seed': 42}
        xg = xgb.XGBClassifier(**all_parameters)
        xg.fit(x_train,
              y_train,
              verbose=0, # set to 1 to see xgb training round intermediate results
              eval_set=[(x_train, y_train), (x_val, y_val)])
        s = xg.score(x_test, y_test)
        max_depth.append(max+4)
        min_child.append(min+4)
        gamma.append(0.1*gam)
        learning_rate.append(0.05*(learn+1))
        score.append(s)
        print('score:', s, 'max_depth:', max + 4, 'min:', min + 4, 'gamma:', 0.1 * gam, 'learn:', 0.05 * (learn + 1))

try:
    np.savetxt('isco68xgbpara.txt',
           np.concatenate((np.array(score)[:,np.newaxis],np.array(max_depth)[:,np.newaxis],np.array(min_child)[:,np.newaxis],
                           np.array(gamma)[:,np.newaxis],np.array(learning_rate)[:,np.newaxis]),axis=1),
           fmt = '%f')
finally:
    print('\n------------------ isco88 xgb -----------------\n')

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

le2 = LabelEncoder()
labels = le2.fit_transform(isco88_data['label'])
x_train, x_test, x_val, y_train, y_test, y_val = data_preprocess.SplitDataset(embeddings, labels, 0.6, 0.3)

num_class = np.unique(labels)
times = range(3)
max_depth = []
min_child = []
gamma = []
learning_rate = []
score = []

for learn in times:
  for gam in times:
    for min in times:
      for max in times:
        all_parameters = {'objective': 'multi:softmax',
                    'num_class': num_class,
                    'gamma': 0.1*gam,
                    'learning_rate': 0.05*(learn+1),
                    'n_estimators': 500,
                    'max_depth': max+4,
                    'min_child_weight': min+4,
                    'early_stopping_rounds': 10,
                    'scale_pos_weight': 1,
                    'tree_method': 'gpu_hist',
                    'eval_metric': ['merror','mlogloss'],
                    'seed': 42}
        xg = xgb.XGBClassifier(**all_parameters)
        xg.fit(x_train,
              y_train,
              verbose=0, # set to 1 to see xgb training round intermediate results
              eval_set=[(x_train, y_train), (x_val, y_val)])
        s = xg.score(x_test, y_test)
        max_depth.append(max+4)
        min_child.append(min+4)
        gamma.append(0.1*gam)
        learning_rate.append(0.05*(learn+1))
        score.append(s)
        print('score:', s, 'max_depth:', max + 4, 'min:', min + 4, 'gamma:', 0.1 * gam, 'learn:', 0.05 * (learn + 1))

try:
    np.savetxt('isco88xgbpara.txt',
           np.concatenate((np.array(score)[:,np.newaxis],np.array(max_depth)[:,np.newaxis],np.array(min_child)[:,np.newaxis],
                           np.array(gamma)[:,np.newaxis],np.array(learning_rate)[:,np.newaxis]),axis=1),
           fmt = '%f')
finally:
    print('\n------------------ end -----------------\n')