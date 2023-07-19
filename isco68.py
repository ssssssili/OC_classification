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

"""
lifew = pd.read_csv('data/(Dutch - ISCO-68) AMIGO_t - Copy.csv', encoding='latin-1')

isco68_short = lifew[lifew['obsseqnr'].astype(str).str.len()==4]

pcs_short = const[~const['code_pcs'].apply(lambda x: type(x) is float or x.find('#') != -1)]

isco68_prep = data_preprocess.PrepData(isco68_short, column=['bjobcoder', 'bwhsID', 'bjobnm', 'bjobdes',
       'bjobco', 'bjobcode', 'bjobcertain'], lan='dutch', lower=True, punc=True, stop_word=True, stemming=True)
isco68_prep.to_csv('isco68_prep.csv', index=False)
"""

isco68_prep = pd.read_csv('isco68_prep.csv')

le = preprocessing.LabelEncoder()
isco68_prep['label'] = le.fit_transform(isco68_prep['obsseqnr'])

isco68_feature = data_preprocess.CombineFeature(isco68_prep, column=['bjobnm', 'bjobdes', 'bjobcertain'], withname= True)

#isco68_namefeature = data_preprocess.CombineFeature(isco68_prep, column=['bjobnm', 'bjobdes', 'bjobcertain'], withname= True)

#data_preprocess.PlotData(isco68_prep, column='obsseqnr')
isco68_data = isco68_feature[['feature', 'label']]

embedding_model = data_preprocess.EmbeddingModel("pdelobelle/robbert-v2-dutch-base")
batch_size = 256
num_batches = len(isco68_data) // batch_size + 1
embeddings = []
labels = []

for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = start_idx + batch_size

    batch_texts = isco68_data['feature'][start_idx:end_idx]
    labels.extend(isco68_data['label'][start_idx:end_idx])

    batch_embeddings = embedding_model.sentence_embedding(batch_texts)
    embeddings.append(batch_embeddings)

embeddings = np.concatenate(embeddings, axis=0)
labels = np.array(labels)

print(embeddings.shape)

all_parameters = {'objective': 'multi:softmax',
                    'num_class': len(np.unique(isco68_data['label'])),
                    'gamma': 0,
                    'learning_rate': 0.1,
                    'n_estimators': 500,
                    'max_depth': 5,
                    'early_stopping_rounds': 10,
                    'scale_pos_weight': 5,
                    'tree_method': 'gpu_hist',
                    'eval_metric': ['merror','mlogloss'],
                    'seed': 42}

x_train, x_test, x_val, y_train, y_test, y_val = data_preprocess.splitDataset(embeddings, labels, 0.6, 0.2)
print(x_train.shape)
print(len(y_train))


xgbtree = xgb.XGBClassifier(**all_parameters)

xgbtree.fit(x_train,
            y_train,
            verbose=1, # set to 1 to see xgb training round intermediate results
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

"""
x = isco68_feature['feature']
y = isco68_feature['label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42)

embedding_model = data_preprocess.EmbeddingModel("pdelobelle/robbert-v2-dutch-base")
# Define batch size for processing
batch_size = 128

num_batches1 = len(x_train) // batch_size + 1
embeddings1 = []
labels1 = []

for i in range(num_batches1):
    start_idx = i * batch_size
    end_idx = start_idx + batch_size

    batch_texts = x_train[start_idx:end_idx]
    labels1.extend(y_train[start_idx:end_idx])

    batch_embeddings = embedding_model.sentence_embedding(batch_texts)
    embeddings1.append(batch_embeddings)

embeddings1 = np.concatenate(embeddings1, axis=0)
labels1 = np.array(labels1)


num_test_batches1 = len(x_test) // batch_size + 1
test_embedding1 = []
test_labels1 = []

for i in range(num_test_batches1):
    start_idx = i * batch_size
    end_idx = start_idx + batch_size

    batch_texts = x_test[start_idx:end_idx]
    test_labels1.extend(y_test[start_idx:end_idx])

    batch_embeddings = embedding_model.sentence_embedding(batch_texts)
    test_embedding1.append(batch_embeddings)

test_embedding1 = np.concatenate(test_embedding1, axis=0)
test_labels1 = np.array(test_labels1)

num_val_batches1 = len(x_val) // batch_size + 1
val_embedding1 = []
val_labels1 = []

for i in range(num_val_batches1):
    start_idx = i * batch_size
    end_idx = start_idx + batch_size

    batch_texts = x_val[start_idx:end_idx]
    val_labels1.extend(y_val[start_idx:end_idx])

    batch_embeddings = embedding_model.sentence_embedding(batch_texts)
    val_embedding1.append(batch_embeddings)

val_embedding1 = np.concatenate(val_embedding1, axis=0)
val_labels1 = np.array(val_labels1)
x_train, y_train = data_preprocess.aggdata(isco68_feature, embeddings1, labels1)
x_val, y_val = data_preprocess.aggdata(isco68_feature, val_embedding1, val_labels1)
"""

"""
for i in range(num_test_batches2):
    start_idx = i * batch_size
    end_idx = start_idx + batch_size

    batch_texts = texts_test2[start_idx:end_idx]
    batch_embeddings = embedding_model.sentence_embedding(batch_texts)

    test_embedding2.append(batch_embeddings)

test_embedding2 = np.concatenate(test_embedding2, axis=0)
X_test2 = xgb.DMatrix(test_embedding2, label=labels_test2)
"""

"""
evals_result1 = {}
evals_result2 = {}

# Train the XGBoost model
params = {
    #'objective': 'multi:softprob',
    'objective': 'multi:softprob',
    'num_class': np.unique(y),
    'eval_metric': 'mlogloss',
    'seed': 42,
    'tree_method': 'gpu_hist',
    'min_child_weight': 1,
    'max_depth': 3,
    'scale_pos_weight': 1,
    'max_delta_step': 0,
    'gamma': 0,
    'eta': 0.1
}

model1 = xgb.XGBClassifier(objective='multi:softmax',
                            num_class=len(np.unique(y)),
                            gamma=0,
                            learning_rate=0.1,
                            n_estimators=500,
                            #missing=0,
                            max_depth=5,
                            reg_lambda=1,
                            #early_stopping_rounds=10,
                            scale_pos_weight= 5,
                            tree_method= 'gpu_hist',
                            eval_metric=['merror','mlogloss'],
                            seed=42)

model1.fit(x_train,
            y_train,
            verbose=0, # set to 1 to see xgb training round intermediate results
            eval_set=[(x_train, y_train), (x_val, y_val)])

results = model1.evals_result()
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

y_pred = model1.predict(test_embedding1)
#print(y_pred)
#np.savetxt('y_pred.txt', y_pred)
#y_pred = model1.predict(x_train)
#np.savetxt('y_pred.txt', y_pred)
#np.savetxt('y_train.txt', y_train)
print('\n------------------ Confusion Matrix -----------------\n')

print('\nAccuracy: {:.2f}'.format(accuracy_score(test_labels1, y_pred)))
"""
"""
print('Micro Precision: {:.2f}'.format(precision_score(test_labels1, y_pred, average='micro')))
print('Micro Recall: {:.2f}'.format(recall_score(test_labels1, y_pred, average='micro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(test_labels1, y_pred, average='micro')))
print('Cohens Kappa: {:.2f}\n'.format(cohen_kappa_score(test_labels1, y_pred)))

print('\n--------------- Classification Report ---------------\n')
print(classification_report(test_labels1, y_pred))
print('---------------------- XGBoost ----------------------') # unnecessary fancy styling
"""
"""
num_rounds = 100
model1 = xgb.train(params, X_train1,
                   evals=[(X_train1, 'Train'), (X_test1, 'Valid')],
                   num_boost_round=num_rounds,
                   evals_result=evals_result1,
                   verbose_eval=True,
                   early_stopping_rounds=10)

print(evals_result1)

train_loss1=list(evals_result1['Train'].values())[0]
valid_loss1=list(evals_result1['Valid'].values())[0]
x_scale1=[i for i in range(len(train_loss1))]
plt.figure(figsize=(10,10))
plt.title('loss1')
plt.plot(x_scale1,train_loss1,label='train',color='r')
plt.plot(x_scale1,valid_loss1,label='valid',color='b')
plt.legend()
plt.savefig('loss_result_1.png')

model2 = xgb.train(params, X_train2,
                   evals=[(X_train2, 'Train'), (X_test2, 'Valid')],
                   num_boost_round=num_rounds,
                   evals_result=evals_result2,
                   verbose_eval=True,
                   early_stopping_rounds=10)
print(evals_result2)

train_loss2=list(evals_result2['Train'].values())[0]
valid_loss2=list(evals_result2['Valid'].values())[0]
x_scale2=[i for i in range(len(train_loss2))]
plt.figure(figsize=(10,10))
plt.title('loss2')
plt.plot(x_scale2,train_loss2,label='train',color='r')
plt.plot(x_scale2,valid_loss2,label='valid',color='b')
plt.legend()
plt.savefig('loss_result_2.png')

# Perform inference on the test set
predictions1 = model1.predict(X_test1)
predictions1 = np.argmax(predictions1, axis=0)
np.savetxt('predictions_1.txt', predictions1, delimiter=',')
predictions2 = model2.predict(X_test2)
predictions2 = np.argmax(predictions2, axis=0)
np.savetxt('predictions_2.txt', predictions2, delimiter=',')
# Calculate accuracy
accuracy1 = accuracy_score(labels_test1, predictions1)
accuracy2 = accuracy_score(labels_test2, predictions2)
f1score1 = f1_score(labels_test1, predictions1, average='micro')
f1score2 = f1_score(labels_test2, predictions2, average='micro')
cohen1 = cohen_kappa_score(labels_test1, predictions1)
cohen2 = cohen_kappa_score(labels_test2, predictions2)
print("first: Accuracy, f1, cohen:", accuracy1, f1score1, cohen1)
print("second: Accuracy, f1, cohen:", accuracy2, f1score2, cohen2)

"""

"""
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
    eval_metric= 'mlogloss',
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
gsearch.fit(X_train2,labels_train2,early_stopping_rounds=10)
print('gsearch1.grid_scores_', gsearch.grid_scores_)
print('gsearch1.best_params_', gsearch.best_params_)
print('gsearch1.best_score_', gsearch.best_score_)

# 尝试param_test1后获得表现最好的参数，然后使用这些参数尝试param_test2
# 获得最好参数组合后降低学习率至0.03尝试

#le.inverse_transform()

"""