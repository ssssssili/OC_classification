from finetuneBert import train_and_evaluate_series_model
import data_preprocess
import pandas as pd

pcs_prep = pd.read_csv('../data/pcs_prep.csv')
pcs_data = data_preprocess.CombineFeature(pcs_prep, column=['numep', 'profession_txt', 'secteur_txt'], withname= False)
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
    [0, 11],            # Unfreeze the classifier and last layer of BERT
    list(range(12)),    # Unfreeze all layers of BERT
    [0, 5, 11]          # Unfreeze the classifier and selected middle layers
]

# Perform training and evaluation for BERT base model
bert_results = train_and_evaluate_series_model(pcs_data['feature'], pcs_data['label'],
                                model_type='bert-base-uncased', layer_configs=layer_configs,
                                batch_size=2, num_epochs=50, max_length=132, num_labels=495, name="pcs",
                                result_filename='result/pcs_bert_results.txt',
                                test_labels_filename='result/pcs_bert_test_labels.txt',
                                test_predictions_filename='result/pcs_bert_test_predictions.txt')

# Perform training and evaluation for multilingual BERT model
multilingual_bert_results = train_and_evaluate_series_model(pcs_data['feature'], pcs_data['label'],
                                model_type='bert-base-multilingual-uncased', layer_configs=layer_configs,
                                batch_size=2, num_epochs=50, max_length=132, num_labels=495, name="pcs",
                                result_filename='result/pcs_mulbert_results.txt',
                                test_labels_filename='result/pcs_mulbert_test_labels.txt',
                                test_predictions_filename='result/pcs_mulbert_test_predictions.txt')


# Print or analyze the results for BERT base model
for config, result in bert_results.items():
    print(f"Configuration: {config}")
    print("Test Accuracy:", result['accuracy'])
    print("Test Precision:", result['precision'])
    print("Test Recall:", result['recall'])
    print("Test F1 Score:", result['f1_score'])
    print("Test Cohen's Kappa:", result['cohen_kappa'])
    print("-----------------------------")

# Print or analyze the results for multilingual BERT model
for config, result in multilingual_bert_results.items():
    print(f"Configuration: {config}")
    print("Test Accuracy:", result['accuracy'])
    print("Test Precision:", result['precision'])
    print("Test Recall:", result['recall'])
    print("Test F1 Score:", result['f1_score'])
    print("Test Cohen's Kappa:", result['cohen_kappa'])
    print("-----------------------------")
