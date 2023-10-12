from finetuneBert import train_and_evaluate_series_model
import data_preprocess
import pandas as pd

asial = pd.read_csv('../data/(English - ISCO-88) AL_allcodes(AsiaLymph) - Copy.csv')
asial_prep = asial[~asial['isco88_cg_4'].str.contains('z')]
isco88_data = data_preprocess.CombineFeature(asial_prep, column=['occupation_en', 'task_en', 'employer_en', 'product_en'], withname=False)
isco88_data['label'] = isco88_data['isco88_cg_4']
isco88_data = isco88_data[['feature', 'label']]

layer_configs = [
    [],                                 # No layer frozen
    ['classifier', 'pooler']            # Unfreeze only the classifier layer
]

mul = 'bert-base-multilingual-cased'
eng = 'bert-base-cased'

mul_results = train_and_evaluate_series_model(isco88_data['feature'], isco88_data['label'],
                                model_type=mul, layer_configs=layer_configs,
                                batch_size=16, num_epochs=100, max_length=305, num_labels=388, name="isco88mul",
                                result_filename='result/isco88_mul_results.txt')

# cased bert
results = train_and_evaluate_series_model(isco88_data['feature'], isco88_data['label'],
                                model_type=eng, layer_configs=layer_configs,
                                batch_size=16, num_epochs=100, max_length=305, num_labels=388, name="isco88eng",
                                result_filename='result/isco88_eng_results.txt')


print("*"*10, "Summary of results", "*"*10)
for config, result in mul_results.items():
    print(f"Configuration: multilingual_uncased_{config}")
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

