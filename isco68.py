import pandas as pd
import numpy as np
import data_preprocess
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing


lifew = pd.read_csv('data/(Dutch - ISCO-68) AMIGO_t - Copy.csv', encoding='latin-1')

isco68_short = lifew[lifew['obsseqnr'].astype(str).str.len()==4]

#pcs_short = const[~const['code_pcs'].apply(lambda x: type(x) is float or x.find('#') != -1)]

#isco68_prep = data_preprocess.PrepData(isco68_short, column=['bjobcoder', 'bwhsID', 'bjobnm', 'bjobdes',
 #      'bjobco', 'bjobcode', 'bjobcertain'], lan='dutch', lower=True, punc=True, stop_word=True, stemming=True)
#print(isco68_prep.head())
#isco68_prep.to_csv('isco68_prep.csv', index=False)

isco68_prep = pd.read_csv('isco68_prep.csv')

le = preprocessing.LabelEncoder()
isco68_prep['label'] = le.fit_transform(isco68_short['obsseqnr'])

isco68_feature = data_preprocess.CombineFeature(isco68_prep, column=['bjobcoder', 'bwhsID', 'bjobnm', 'bjobdes',
                                                                 'bjobco', 'bjobcode', 'bjobcertain'], withname= False)

isco68_namefeature = data_preprocess.CombineFeature(isco68_prep, column=['bjobcoder', 'bwhsID', 'bjobnm', 'bjobdes',
                                                                 'bjobco', 'bjobcode', 'bjobcertain'], withname= True)

#data_preprocess.PlotData(isco68_prep, column='obsseqnr')

x = isco68_feature['feature']
y = isco68_feature['label']

x_name = isco68_namefeature['feature']
y_name = isco68_namefeature['label']

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

num_class = len(isco68_feature['label'].value_counts())

# Train the XGBoost model
params = {
    'objective': 'multi:softmax',
    'num_class': num_class,
    'eval_metric': 'error',
    'seed': 42,
    'tree_method': 'gpu_hist',
    'min_child_weight': 1,
    'max_depth': 3,
    'scale_pos_weight': 1,
    'max_delta_step': 0,
    'gamma': 0,
    'eta': 0.1
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

predictions1 = np.argmax(predictions1, axis=1)

predictions2 = np.argmax(predictions2, axis=1)

# Calculate accuracy
accuracy1 = accuracy_score(labels_test1, predictions1)
accuracy2 = accuracy_score(labels_test2, predictions2)
f1score1 = f1_score(labels_test1, predictions1)
f1score2 = f1_score(labels_test2, predictions2)
cohen1 = cohen_kappa_score(labels_test1, predictions1)
cohen2 = cohen_kappa_score(labels_test2, predictions2)

print("first: Accuracy, f1, cohen:", accuracy1)
print("second: Accuracy, f1, cohen:", accuracy2)


param_test1 = {
'max_depth':[i for i in range(3,10,2)],
'min_child_weight':[i for i in range(1,6,2)]
}

param_test2 = {
 'gamma':[i/10.0 for i in range(0,5)]
}

gsearch = GridSearchCV(
estimator = XGBClassifier(
    objective= 'multi:softmax',
    num_class= num_class,
    eval_metric= 'error',
    n_estimators= 140,
    seed= 42,
    tree_method= 'gpu_hist',
    min_child_weight= 1,
    max_depth= 3,
    scale_pos_weight= 1,
    max_delta_step= 0,
    gamma= 0,
    eta= 0.1),
    param_grid = param_test1,
    n_jobs=4,
    iid=False,
    cv=5)


# 我假设使用第二组数据效果比较好，如果使用第一组效果更好的话把下列的2都改成1就行
gsearch.fit(X_train2,labels_train2)
print('gsearch1.grid_scores_', gsearch.grid_scores_)
print('gsearch1.best_params_', gsearch.best_params_)
print('gsearch1.best_score_', gsearch.best_score_)

# 尝试param_test1后获得表现最好的参数，然后使用这些参数尝试param_test2
# 获得最好参数组合后降低学习率至0.03尝试

#le.inverse_transform()