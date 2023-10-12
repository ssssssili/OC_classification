from finetuneBert import train_and_evaluate_series_model
import data_preprocess
import pandas as pd

const = pd.read_csv('../data/(French - PCS-NAF) Constances_CPro_MG_Operas_012021.csv')
const[['code_naf','code_pcs']].dropna(axis=0,how='any')
pcs_prep = const[const['code_pcs'].str.contains('#') == False]
pcs_data = data_preprocess.CombineFeature(pcs_prep, column=['profession_txt', 'secteur_txt'], withname= False)
pcs_data['label'] = pcs_data['code_pcs']
pcs_data = pcs_data[['feature', 'label']]

layer_configs = [
    [],                                 # No layer frozen
    ['classifier', 'pooler']            # Unfreeze only the classifier layer
]

mul = 'bert-base-multilingual-cased'
fr = 'dbmdz/bert-base-french-europeana-cased'

mul_results = train_and_evaluate_series_model(pcs_data['feature'], pcs_data['label'],
                                model_type=mul, layer_configs=layer_configs,
                                batch_size=16, num_epochs=100, max_length=132, num_labels=495, name="pcsmul",
                                result_filename='result/pcs_mul_results.txt')

bert_results = train_and_evaluate_series_model(pcs_data['feature'], pcs_data['label'],
                                model_type=fr, layer_configs=layer_configs,
                                batch_size=16, num_epochs=100, max_length=132, num_labels=495, name="pcsfr",
                                result_filename='result/pcs_fr_results.txt')


print("*"*10, "Summary of results", "*"*10)
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