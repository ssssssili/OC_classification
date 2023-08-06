from finetuneBert import train_and_evaluate_series_model
import data_preprocess
import pandas as pd

naf_prep = pd.read_csv('../data/naf_prep.csv')
naf_data = data_preprocess.CombineFeature(naf_prep, column=['numep', 'profession_txt', 'secteur_txt'], withname= False)
naf_data['label'] = naf_data['code_naf']
naf_data = naf_data[['feature', 'label']]

"""
maxnum = []
for i in naf_data['feature']:
    cnt = 0
    for j in i:
        cnt += 1
    maxnum.append(cnt)
print(max(maxnum))
print(len(naf_data['label'].value_counts()))
"""

layer_configs = [
    [0],                # Unfreeze only the classifier layer
    [0, 11],            # Unfreeze the classifier and last layer of BERT
    list(range(12)),    # Unfreeze all layers of BERT
    [0, 5, 11]          # Unfreeze the classifier and selected middle layers
]

# Perform training and evaluation for multilingual BERT model
multilingual_bert_results = train_and_evaluate_series_model(naf_data['feature'], naf_data['label'],
                                model_type='bert-base-multilingual-uncased', layer_configs=layer_configs,
                                batch_size=2, num_epochs=10, max_length=132, num_labels=732, name="nafmul",
                                result_filename='result/naf_mulbert_results.txt',
                                test_labels_filename='result/naf_mulbert_test_labels.txt',
                                test_predictions_filename='result/naf_mulbert_test_predictions.txt')


# Print or analyze the results for multilingual BERT model
for config, result in multilingual_bert_results.items():
    print(f"Configuration: {config}")
    print("Test Accuracy:", result['accuracy'])
    print("Test Precision:", result['precision'])
    print("Test Recall:", result['recall'])
    print("Test F1 Score:", result['f1_score'])
    print("Test Cohen's Kappa:", result['cohen_kappa'])
    print("-----------------------------")
