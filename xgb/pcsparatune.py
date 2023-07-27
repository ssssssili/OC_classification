import pandas as pd
import numpy as np
import data_preprocess
import xgboost as xgb
import time
from sklearn.preprocessing import LabelEncoder
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

pcs_prep = pd.read_csv('../data/pcs_prep.csv')
pcs_data = data_preprocess.CombineFeature(pcs_prep, column=['numep', 'profession_txt', 'secteur_txt'], withname= False)
pcs_data['label'] = pcs_data['code_pcs']
pcs_data = pcs_data[['feature', 'label']]

embedding_model = data_preprocess.EmbeddingModel("benjamin/roberta-base-wechsel-french")
batch_size = 256
num_batches = len(pcs_data) // batch_size + 1
embeddings = []
labels = []

for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = start_idx + batch_size
    batch_texts = pcs_data['feature'][start_idx:end_idx]
    batch_embeddings = embedding_model.sentence_embedding(batch_texts)
    embeddings.append(batch_embeddings)
embeddings = np.concatenate(embeddings, axis=0)

print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
print('\n------------------ pcs xgb -----------------\n')
le1 = LabelEncoder()
labels = le1.fit_transform(pcs_data['label'])
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
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        all_parameters = {'objective': 'multi:softmax',
                    'num_class': num_class,
                    'gamma': 0.1*gam,
                    'learning_rate': 0.05*(learn+1),
                    'n_estimators': 500,
                    'max_depth': max+4,
                    'min_child_weight': min+4,
                    'early_stopping_rounds': 10,
                    #'scale_pos_weight': 1,
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
    np.savetxt('pcsxgbpara.txt',
           np.concatenate((np.array(score)[:,np.newaxis],np.array(max_depth)[:,np.newaxis],np.array(min_child)[:,np.newaxis],
                           np.array(gamma)[:,np.newaxis],np.array(learning_rate)[:,np.newaxis]),axis=1),
           fmt = '%f')
except:
    print('error when saving file')

print('\n------------------ end -----------------\n')