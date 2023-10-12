from finetuneBert import train_and_evaluate_series_model
import data_preprocess
import pandas as pd
import numpy as np

"""
layer_configs = [
    ['classifier', 'pooler'],                       # Unfreeze only the classifier layer
    ['classifier', 'pooler', 'layer.11'],            # Unfreeze the classifier and last layer of BERT
    ['classifier', 'pooler', 'layer.11', 'layer.10'],
    ['classifier', 'pooler', 'layer.11', 'layer.7'],
    ['classifier', 'pooler', 'layer.11', 'layer.7', 'layer.0']
]

print('*'*10, 'isco68', '*'*10)

lifew = pd.read_csv('../data/(Dutch - ISCO-68) AMIGO_t - Copy.csv', encoding='latin-1')
lifew_prep = lifew[lifew['bjobcode'].astype(str).str.len()==7]
isco68_data = data_preprocess.CombineFeature(lifew_prep, column=['bjobnm'], withname= False)
isco68_data['label'] = isco68_data['bjobcode']
isco68_data = isco68_data[['feature', 'label']]

multilingual_uncased = 'bert-base-multilingual-uncased'
multilingual_cased = 'bert-base-multilingual-cased'
monolingual_cased = 'GroNLP/bert-base-dutch-cased'

# Preprocess uncased text
isco68_data_un = data_preprocess.PrepData(isco68_data, column=['feature'], lan='dutch', lower=True, punc=True, stop_word=False, stemming=False)

# multilingual uncased bert
mul_un_results = train_and_evaluate_series_model(isco68_data_un['feature'], isco68_data_un['label'],
                                model_type=multilingual_uncased, layer_configs=layer_configs,
                                batch_size=8, num_epochs=100, max_length=100, num_labels=639, name="isco68mulun",
                                result_filename='result/isco68_mulun_results.txt',
                                test_labels_filename='result/isco68_mulun_test_labels.txt',
                                test_predictions_filename='result/isco68_mulun_test_predictions.txt')

# Preprocess cased text
isco68_data_cased = data_preprocess.PrepData(isco68_data, column=['feature'], lan='dutch', lower=False, punc=True, stop_word=False, stemming=False)

# multilingual cased bert
mul_results = train_and_evaluate_series_model(isco68_data_cased['feature'], isco68_data_cased['label'],
                                model_type=multilingual_cased, layer_configs=layer_configs,
                                batch_size=8, num_epochs=100, max_length=100, num_labels=639, name="isco68mul",
                                result_filename='result/isco68_mul_results.txt',
                                test_labels_filename='result/isco68_mul_test_labels.txt',
                                test_predictions_filename='result/isco68_mul_test_predictions.txt')

# cased bert
bert_results = train_and_evaluate_series_model(isco68_data_cased['feature'], isco68_data_cased['label'],
                                model_type=monolingual_cased, layer_configs=layer_configs,
                                batch_size=8, num_epochs=100, max_length=100, num_labels=639, name="isco68",
                                result_filename='result/isco68_results.txt',
                                test_labels_filename='result/isco68_test_labels.txt',
                                test_predictions_filename='result/isco68_test_predictions.txt')

print("*"*10, "Summary of results", "*"*10)
for config, result in mul_un_results.items():
    print(f"Configuration: multilingual_uncased_{config}")
    print("Test Accuracy:", result['accuracy'])
    print("Test Precision:", result['precision'])
    print("Test Recall:", result['recall'])
    print("Test F1 Score:", result['f1_score'])
    print("Test Cohen's Kappa:", result['cohen_kappa'])
    print("-----------------------------")

for config, result in mul_results.items():
    print(f"Configuration: mul_results_{config}")
    print("Test Accuracy:", result['accuracy'])
    print("Test Precision:", result['precision'])
    print("Test Recall:", result['recall'])
    print("Test F1 Score:", result['f1_score'])
    print("Test Cohen's Kappa:", result['cohen_kappa'])
    print("-----------------------------")

for config, result in bert_results.items():
    print(f"Configuration: bert_results_{config}")
    print("Test Accuracy:", result['accuracy'])
    print("Test Precision:", result['precision'])
    print("Test Recall:", result['recall'])
    print("Test F1 Score:", result['f1_score'])
    print("Test Cohen's Kappa:", result['cohen_kappa'])
    print("-----------------------------")

print('*'*10, 'pcs', '*'*10)
"""
const = pd.read_csv('../data/(French - PCS-NAF) Constances_CPro_MG_Operas_012021.csv')
const[['code_naf','code_pcs']].dropna(axis=0,how='any')
pcs_prep = const[const['code_pcs'].str.contains('#') == False]
pcs_data = data_preprocess.CombineFeature(pcs_prep, column=['profession_txt', 'secteur_txt'], withname= False)
pcs_data['label'] = pcs_data['code_pcs']
pcs_data = pcs_data[['feature', 'label']]

print(len(pcs_data))
exit()

print(len(pcs_data['label'].value_counts()), pcs_data['label'].value_counts().mean(), pcs_data['label'].value_counts().std())
print(max(pcs_data['label'].value_counts()), min(pcs_data['label'].value_counts()))
maxnum = []
for i in pcs_data['feature']:
    cnt = 0
    for j in i:
        cnt += 1
    maxnum.append(cnt)
print(max(maxnum), min(maxnum))
print(np.mean(maxnum), np.std(maxnum))
print(pd.Series(maxnum).mean(),pd.Series(maxnum).std())
print(max(pd.Series(maxnum).value_counts()))
data_preprocess.PlotData(pcs_data['label'])
data_preprocess.PlotData(pd.Series(maxnum))
exit()

"""
maxnum = []
for i in pcs_data['feature']:
    cnt = 0
    for j in i:
        cnt += 1
    maxnum.append(cnt)
print(max(maxnum))
print(len(pcs_data['label'].value_counts()))
"""

multilingual_uncased = 'bert-base-multilingual-uncased'
multilingual_cased = 'bert-base-multilingual-cased'
monolingual_cased = 'dbmdz/bert-base-french-europeana-cased'

# Preprocess uncased text
pcs_data_un = data_preprocess.PrepData(pcs_data, column=['feature'],
                                    lan='french', lower=True, punc=True, stop_word=False, stemming=False)

mul_un_results = train_and_evaluate_series_model(pcs_data_un['feature'], pcs_data_un['label'],
                                model_type=multilingual_uncased, layer_configs=layer_configs,
                                batch_size=8, num_epochs=100, max_length=132, num_labels=495, name="pcsmulun",
                                result_filename='result/pcs_mulun_results.txt',
                                test_labels_filename='result/pcs_mulun_test_labels.txt',
                                test_predictions_filename='result/pcs_mulun_test_predictions.txt')

# Preprocess cased text
pcs_data_cased = data_preprocess.PrepData(pcs_data, column=['feature'],
                                    lan='french', lower=False, punc=True, stop_word=False, stemming=False)

mul_results = train_and_evaluate_series_model(pcs_data_cased['feature'], pcs_data_cased['label'],
                                model_type=multilingual_cased, layer_configs=layer_configs,
                                batch_size=8, num_epochs=100, max_length=132, num_labels=495, name="pcsmul",
                                result_filename='result/pcs_mul_results.txt',
                                test_labels_filename='result/pcs_mul_test_labels.txt',
                                test_predictions_filename='result/pcs_mul_test_predictions.txt')

bert_results = train_and_evaluate_series_model(pcs_data_cased['feature'], pcs_data_cased['label'],
                                model_type=monolingual_cased, layer_configs=layer_configs,
                                batch_size=8, num_epochs=100, max_length=132, num_labels=495, name="pcs",
                                result_filename='result/pcs_results.txt',
                                test_labels_filename='result/pcs_test_labels.txt',
                                test_predictions_filename='result/pcs_test_predictions.txt')


print("*"*10, "Summary of results", "*"*10)
for config, result in mul_un_results.items():
    print(f"Configuration: multilingual_uncased_{config}")
    print("Test Accuracy:", result['accuracy'])
    print("Test Precision:", result['precision'])
    print("Test Recall:", result['recall'])
    print("Test F1 Score:", result['f1_score'])
    print("Test Cohen's Kappa:", result['cohen_kappa'])
    print("-----------------------------")

for config, result in mul_results.items():
    print(f"Configuration: mul_results_{config}")
    print("Test Accuracy:", result['accuracy'])
    print("Test Precision:", result['precision'])
    print("Test Recall:", result['recall'])
    print("Test F1 Score:", result['f1_score'])
    print("Test Cohen's Kappa:", result['cohen_kappa'])
    print("-----------------------------")

for config, result in bert_results.items():
    print(f"Configuration: bert_results_{config}")
    print("Test Accuracy:", result['accuracy'])
    print("Test Precision:", result['precision'])
    print("Test Recall:", result['recall'])
    print("Test F1 Score:", result['f1_score'])
    print("Test Cohen's Kappa:", result['cohen_kappa'])
    print("-----------------------------")