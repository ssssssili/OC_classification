from finetuneBert import train_and_evaluate_series_model
import data_preprocess
import pandas as pd

const = pd.read_csv('../data/(French - PCS-NAF) Constances_CPro_MG_Operas_012021.csv')
const[['code_naf','code_pcs']].dropna(axis=0,how='any')
pcs_prep = const[const['code_pcs'].str.contains('#') == False]
pcs_data = data_preprocess.CombineFeature(pcs_prep, column=['profession_txt', 'secteur_txt'], withname= False)
pcs_data['label'] = pcs_data['code_pcs']
pcs_data = pcs_data[['feature', 'label']]

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

layer_configs = [
    [0],                # Unfreeze only the classifier layer
#    [0, 11],            # Unfreeze the classifier and last layer of BERT
#    list(range(12)),    # Unfreeze all layers of BERT
    [0, 7, 11]          # Unfreeze the classifier and selected middle layers
]

multilingual_uncased = 'bert-base-multilingual-uncased'
multilingual_cased = 'bert-base-multilingual-cased'
monolingual_cased = 'dbmdz/bert-base-french-europeana-cased'

# Preprocess uncased text
pcs_data_un = data_preprocess.PrepData(pcs_data, column=['feature'],
                                    lan='french', lower=True, punc=True, stop_word=False, stemming=False)

mul_un_results = train_and_evaluate_series_model(pcs_data_un['feature'], pcs_data_un['label'],
                                model_type=multilingual_uncased, layer_configs=layer_configs,
                                batch_size=2, num_epochs=100, max_length=132, num_labels=495, name="pcsmulun",
                                result_filename='result/pcs_mulun_results.txt',
                                test_labels_filename='result/pcs_mulun_test_labels.txt',
                                test_predictions_filename='result/pcs_mulun_test_predictions.txt')

# Preprocess cased text
pcs_data_cased = data_preprocess.PrepData(pcs_data, column=['feature'],
                                    lan='french', lower=False, punc=True, stop_word=False, stemming=False)

mul_results = train_and_evaluate_series_model(pcs_data_cased['feature'], pcs_data_cased['label'],
                                model_type=multilingual_cased, layer_configs=layer_configs,
                                batch_size=2, num_epochs=100, max_length=132, num_labels=495, name="pcsmul",
                                result_filename='result/pcs_mul_results.txt',
                                test_labels_filename='result/pcs_mul_test_labels.txt',
                                test_predictions_filename='result/pcs_mul_test_predictions.txt')

bert_results = train_and_evaluate_series_model(pcs_data_cased['feature'], pcs_data_cased['label'],
                                model_type=monolingual_cased, layer_configs=layer_configs,
                                batch_size=2, num_epochs=100, max_length=132, num_labels=495, name="pcs",
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