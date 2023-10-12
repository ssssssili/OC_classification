from finetuneBert import train_and_evaluate_series_model
import data_preprocess
import pandas as pd

const = pd.read_csv('../data/(French - PCS-NAF) Constances_CPro_MG_Operas_012021.csv')
const[['code_naf','code_pcs']].dropna(axis=0,how='any')
naf_prep = const[const['code_naf'].str.contains('#') == False]
naf_data = data_preprocess.CombineFeature(naf_prep, column=['profession_txt', 'secteur_txt'], withname= False)
naf_data['label'] = naf_data['code_naf']
naf_data = naf_data[['feature', 'label']]

layer_configs = [
    [],                                 # No layer frozen
    ['classifier', 'pooler']            # Unfreeze only the classifier layer
]

mul = 'bert-base-multilingual-cased'
fr = 'dbmdz/bert-base-french-europeana-cased'

mul_results = train_and_evaluate_series_model(naf_data['feature'], naf_data['label'],
                                model_type=mul, layer_configs=layer_configs,
                                batch_size=8, num_epochs=100, max_length=132, num_labels=732, name="nafmul",
                                result_filename='result/naf_mul_results.txt')

results = train_and_evaluate_series_model(naf_data['feature'], naf_data['label'],
                                model_type=fr, layer_configs=layer_configs,
                                batch_size=8, num_epochs=100, max_length=132, num_labels=732, name="naffr",
                                result_filename='result/naf_fr_results.txt')


print("*"*10, "Summary of results", "*"*10)
for config, result in mul_results.items():
    print(f"Configuration: mul_results_{config}")
    print("Test Accuracy:", result['accuracy'])
    print("Test Precision:", result['precision'])
    print("Test Recall:", result['recall'])
    print("Test F1 Score:", result['f1_score'])
    print("Test Cohen's Kappa:", result['cohen_kappa'])
    print("-----------------------------")

for config, result in results.items():
    print(f"Configuration: bert_results_{config}")
    print("Test Accuracy:", result['accuracy'])
    print("Test Precision:", result['precision'])
    print("Test Recall:", result['recall'])
    print("Test F1 Score:", result['f1_score'])
    print("Test Cohen's Kappa:", result['cohen_kappa'])
    print("-----------------------------")