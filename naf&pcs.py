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


const = pd.read_csv('data/(French - PCS-NAF) Constances_CPro_MG_Operas_012021.csv')

pcs_short = const[~const['code_pcs'].apply(lambda x: type(x) is float or x.find('#') != -1)]
naf_short = const[~const['code_naf'].apply(lambda x: type(x) is float or x.find('#') != -1)]

#pcs_prep = data_preprocess.PrepData(pcs_short, column=['numep', 'profession_txt', 'secteur_txt'],
 #                                     lan='french', lower=True, punc=True, stop_word=True, stemming=True)
#print(pcs_prep.head())
#pcs_prep.to_csv('pcs_prep.csv', index=False)

#naf_prep = data_preprocess.PrepData(naf_short, column=['numep', 'profession_txt', 'secteur_txt'],
 #                                     lan='french', lower=True, punc=True, stop_word=True, stemming=True)
#print(naf_prep.head())
#naf_prep.to_csv('naf_prep.csv', index=False)

pcs_prep = pd.read_csv('pcs_prep.csv')
naf_prep = pd.read_csv('naf_prep.csv')

le = preprocessing.LabelEncoder()
pcs_prep['label'] = le.fit_transform(pcs_prep['code_pcs'])
naf_prep['label'] = le.fit_transform(naf_prep['code_naf'])

pcs_feature = data_preprocess.CombineFeature(pcs_prep, column=['numep', 'profession_txt',
                                                                  'secteur_txt'], withname= True)

naf_feature = data_preprocess.CombineFeature(naf_prep, column=['numep', 'profession_txt',
                                                               'secteur_txt'], withname= True)

pcs_data = pcs_feature[['feature', 'label']]
naf_data = naf_feature[['feature', 'label']]

"""
data_preprocess.PlotData(pcs_prep, column='code_pcs')
pcs_index = data_preprocess.DataSplit(pcs_prep, column='code_pcs', thershold=5, num=4)

pcs_1 = pcs_prep.loc[pcs_prep['code_pcs'].isin(pcs_index[0])]
pcs_2 = pcs_prep.loc[pcs_prep['code_pcs'].isin(pcs_index[1])]
pcs_3 = pcs_prep.loc[pcs_prep['code_pcs'].isin(pcs_index[2])]
pcs_4 = pcs_prep.loc[pcs_prep['code_pcs'].isin(pcs_index[3])]

naf_index = data_preprocess.DataSplit(naf_prep, column='code_naf', thershold=5, num=5)

naf_1 = naf_prep.loc[naf_prep['code_naf'].isin(naf_index[0])]
naf_2 = naf_prep.loc[naf_prep['code_naf'].isin(naf_index[1])]
naf_3 = naf_prep.loc[naf_prep['code_naf'].isin(naf_index[2])]
naf_4 = naf_prep.loc[naf_prep['code_naf'].isin(naf_index[3])]
naf_5 = naf_prep.loc[naf_prep['code_naf'].isin(naf_index[4])]

"""