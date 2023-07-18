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

asial = pd.read_csv('../data/(English - ISCO-88) AL_allcodes(AsiaLymph) - Copy.csv')

isco88_short = asial[~asial['isco88_cg_4'].str.contains('z')]

#isco88_prep = data_preprocess.PrepData(isco88_short, column=['occupation_en', 'task_en', 'employer_en', 'product_en'],
 #                                     lan='english', lower=True, punc=True, stop_word=True, stemming=True)
#print(isco88_prep.head())
#isco88_prep.to_csv('isco88_prep.csv', index=False)

isco88_prep = pd.read_csv('../isco88_prep.csv')

le = preprocessing.LabelEncoder()
isco88_prep['label'] = le.fit_transform(isco88_prep['isco88_cg_4'])

isco88_feature = data_preprocess.CombineFeature(isco88_prep, column=['occupation_en', 'task_en', 'employer_en',
                                                                     'product_en'], withname=True)

x = isco88_feature['feature']
y = isco88_feature['label']
isco88_data = isco88_feature[['feature', 'label']]

def splitData(dataset, training, test):
    tem = []
    for l in np.unique(dataset['label']):
        if dataset['label'].value_counts()[l] == 1:
            tem.append(l)

    df = dataset[dataset['label'].apply(lambda x: x not in tem)]
    print(len(tem))
    print(len(df['label'].value_counts()))
    x_train, x_test, y_train, y_test = train_test_split(df['feature'], df['label'], test_size=(1 - training),
                                                        stratify=df['label'], random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=(test / (1 - training)),
                                                    random_state=42)

    for class_label in tem:
        class_index = np.where(dataset['label'] == class_label)[0][0]
        #pd.concat([x_train, pd.Series(dataset.loc[class_index]['feature'])], axis=0)
        #pd.concat([x_train, pd.Series(class_label)], axis=0)
        x_train= x_train.append(pd.Series(dataset.loc[class_index]['feature']))
        y_train = y_train.append(pd.Series(class_label))

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    x_val = np.array(x_val)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_val = np.array(y_val)

    return x_train, x_test, x_val, y_train, y_test, y_val

#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
#x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.14, random_state=42)
x_train, x_test, x_val, y_train, y_test, y_val = splitData(isco88_data, 0.6, 0.3)

embedding_model = data_preprocess.EmbeddingModel("roberta-base")
batch_size = 128

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

xgbtree = xgb.XGBClassifier(objective='multi:softmax',
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
                            seed=42)

xgbtree.fit(embeddings,
            labels,
            verbose=0, # set to 1 to see xgb training round intermediate results
            eval_set=[(embeddings, labels), (val_embedding, val_labels)])

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

y_pred = xgbtree.predict(test_embedding)
print(y_pred)
np.savetxt('y_pred.txt', y_pred)
#y_pred = model1.predict(x_train)
#np.savetxt('y_pred.txt', y_pred)
#np.savetxt('y_train.txt', y_train)
print('\n------------------ Confusion Matrix -----------------\n')

print('\nAccuracy: {:.2f}'.format(accuracy_score(test_labels, y_pred)))
print('Micro Precision: {:.2f}'.format(precision_score(test_labels, y_pred, average='micro')))
print('Micro Recall: {:.2f}'.format(recall_score(test_labels, y_pred, average='micro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(test_labels, y_pred, average='micro')))
print('Cohens Kappa: {:.2f}\n'.format(cohen_kappa_score(test_labels, y_pred)))

print('\n--------------- Classification Report ---------------\n')
print(classification_report(test_labels, y_pred))
print('---------------------- XGBoost ----------------------') 

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


