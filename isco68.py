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

drop = []
for i in np.unique(isco68_prep['label']):
    if isco68_prep['label'].value_counts()[i]==1:
        drop.append(i)

isco68_prep = isco68_prep[isco68_prep['label'].apply(lambda x: x not in drop)]

isco68_feature = data_preprocess.CombineFeature(isco68_prep, column=['bjobcoder', 'bwhsID', 'bjobnm', 'bjobdes',
                                                                 'bjobco', 'bjobcode', 'bjobcertain'], withname= True)

#isco68_namefeature = data_preprocess.CombineFeature(isco68_prep, column=['bjobcoder', 'bwhsID', 'bjobnm', 'bjobdes',
 #                                                                'bjobco', 'bjobcode', 'bjobcertain'], withname= True)

#data_preprocess.PlotData(isco68_prep, column='obsseqnr')

#x = isco68_feature['feature']
#y = isco68_feature['label']
x = isco68_feature['feature']
y = isco68_feature['label']

texts_train1, texts_test1, labels_train1, labels_test1 = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)
#texts_train2, texts_test2, labels_train2, labels_test2 = train_test_split(x_name, y_name, test_size=0.3, random_state=42)


embedding_model = data_preprocess.EmbeddingModel("pdelobelle/robbert-v2-dutch-base")
# Define batch size for processing
batch_size = 128

# Calculate the number of batches
num_batches1 = len(texts_train1) // batch_size + 1
#num_batches2 = len(texts_train2) // batch_size + 1

# Initialize empty arrays for storing embeddings and labels
embeddings1 = []
labels1 = []

#embeddings2 = []
#batch_labels2 = []

# Process data in batches
for i in range(num_batches1):
    start_idx = i * batch_size
    end_idx = start_idx + batch_size

    batch_texts = texts_train1[start_idx:end_idx]
    labels1.extend(labels_train1[start_idx:end_idx])

    batch_embeddings = embedding_model.sentence_embedding(batch_texts)
    embeddings1.append(batch_embeddings)

"""
for i in range(num_batches2):
    start_idx = i * batch_size
    end_idx = start_idx + batch_size

    batch_texts = texts_train2[start_idx:end_idx]
    batch_labels2.extend(labels_train2[start_idx:end_idx])

    batch_embeddings = embedding_model.sentence_embedding(batch_texts)
    embeddings2.append(batch_embeddings)
"""
# Concatenate embeddings and labels
embeddings1 = np.concatenate(embeddings1, axis=0)
labels1 = np.array(labels1)

#embeddings2 = np.concatenate(embeddings2, axis=0)
#labels_train2 = np.array(batch_labels2)

# Convert the embeddings and labels to NumPy arrays
#X_train1 = xgb.DMatrix(embeddings1, label=labels_train1)
#X_train2 = xgb.DMatrix(embeddings2, label=labels_train2)

# Perform inference on the test set

num_test_batches1 = len(texts_test1) // batch_size + 1
test_embedding1 = []
test_labels1 = []
#predictions1 = []

#num_test_batches2 = len(texts_test2) // batch_size + 1
#test_embedding2 = []
#predictions2 = []

for i in range(num_test_batches1):
    start_idx = i * batch_size
    end_idx = start_idx + batch_size

    batch_texts = texts_test1[start_idx:end_idx]
    test_labels1.extend(labels_train1[start_idx:end_idx])

    batch_embeddings = embedding_model.sentence_embedding(batch_texts)
    test_embedding1.append(batch_embeddings)

test_embedding1 = np.concatenate(test_embedding1, axis=0)
test_labels1 = np.array(test_labels1)
#X_test1 = xgb.DMatrix(test_embedding1, label=labels_test1)

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
"""

model1 = xgb.XGBClassifier(objective='multi:softmax',
                            num_class=len(np.unique(y)),
                            gamma=0,
                            learning_rate=0.1,
                            max_depth=5,
                            reg_lambda=1,
                            early_stopping_rounds=10,
                            scale_pos_weight= 5,
                            tree_method= 'gpu_hist',
                            eval_metric=['merror','mlogloss'],
                            seed=42)

model1.fit(embeddings1,
            labels1,
            verbose=0, # set to 1 to see xgb training round intermediate results
            eval_set=[(embeddings1, labels1), (test_embedding1, test_labels1)])

results = model1.evals_result()
epochs = len(results['validation_0']['mlogloss'])
x_axis = range(0, epochs)

# xgboost 'mlogloss' plot
fig, ax = plt.subplots(figsize=(9,5))
ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
ax.plot(x_axis, results['validation_1']['mlogloss'], label='Test')
ax.legend()
plt.ylabel('mlogloss')
plt.title('GridSearchCV XGBoost mlogloss')
plt.show()

# xgboost 'merror' plot
fig, ax = plt.subplots(figsize=(9,5))
ax.plot(x_axis, results['validation_0']['merror'], label='Train')
ax.plot(x_axis, results['validation_1']['merror'], label='Test')
ax.legend()
plt.ylabel('merror')
plt.title('GridSearchCV XGBoost merror')
plt.show()

## ---------- Model Classification Report ----------
## get predictions and create model quality report

y_pred = model1.predict(test_embedding1)

print('\n------------------ Confusion Matrix -----------------\n')
print(confusion_matrix(np.array(labels_test1), y_pred))

print('\nAccuracy: {:.2f}'.format(accuracy_score(np.array(labels_test1), y_pred)))
print('Balanced Accuracy: {:.2f}\n'.format(balanced_accuracy_score(np.array(labels_test1), y_pred)))

print('Micro Precision: {:.2f}'.format(precision_score(np.array(labels_test1), y_pred, average='micro')))
print('Micro Recall: {:.2f}'.format(recall_score(np.array(labels_test1), y_pred, average='micro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(np.array(labels_test1), y_pred, average='micro')))

print('Macro Precision: {:.2f}'.format(precision_score(np.array(labels_test1), y_pred, average='macro')))
print('Macro Recall: {:.2f}'.format(recall_score(np.array(labels_test1), y_pred, average='macro')))
print('Macro F1-score: {:.2f}\n'.format(f1_score(np.array(labels_test1), y_pred, average='macro')))

print('Weighted Precision: {:.2f}'.format(precision_score(np.array(labels_test1), y_pred, average='weighted')))
print('Weighted Recall: {:.2f}'.format(recall_score(np.array(labels_test1), y_pred, average='weighted')))
print('Weighted F1-score: {:.2f}'.format(f1_score(np.array(labels_test1), y_pred, average='weighted')))

print('\n--------------- Classification Report ---------------\n')
print(classification_report(np.array(labels_test1), y_pred))
print('---------------------- XGBoost ----------------------') # unnecessary fancy styling

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