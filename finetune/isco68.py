from finetuneBert import train_and_evaluate_series_model
import data_preprocess
import pandas as pd

isco68_prep = pd.read_csv('../data/isco68_prep.csv')
isco68_data = data_preprocess.CombineFeature(isco68_prep, column=['bjobnm'], withname= False)
isco68_data['label'] = isco68_data['bjobcode']
isco68_data = isco68_data[['feature', 'label']]

"""
maxnum = []
for i in isco68_data['feature']:
    cnt = 0
    for j in i:
        cnt += 1
    maxnum.append(cnt)
print(max(maxnum))
print(len(isco68_data['label'].value_counts()))
"""

layer_configs = [
    [0],                # Unfreeze only the classifier layer
    [0, 11],            # Unfreeze the classifier and last layer of BERT
    list(range(12)),    # Unfreeze all layers of BERT
    [0, 5, 11]          # Unfreeze the classifier and selected middle layers
]

# Perform training and evaluation for BERT base model
bert_results = train_and_evaluate_series_model(isco68_data['feature'], isco68_data['label'],
                                model_type='bert-base-uncased', layer_configs=layer_configs,
                                batch_size=2, num_epochs=20, max_length=100, num_labels=639, name="isco68bert",
                                result_filename='result/isco68_bert_results.txt',
                                test_labels_filename='result/isco68_bert_test_labels.txt',
                                test_predictions_filename='result/isco68_bert_test_predictions.txt')


# Print or analyze the results for BERT base model
for config, result in bert_results.items():
    print(f"Configuration: {config}")
    print("Test Accuracy:", result['accuracy'])
    print("Test Precision:", result['precision'])
    print("Test Recall:", result['recall'])
    print("Test F1 Score:", result['f1_score'])
    print("Test Cohen's Kappa:", result['cohen_kappa'])
    print("-----------------------------")


# Perform training and evaluation for multilingual BERT model
multilingual_bert_results = train_and_evaluate_series_model(isco68_data['feature'], isco68_data['label'],
                                model_type='bert-base-multilingual-uncased', layer_configs=layer_configs,
                                batch_size=2, num_epochs=20, max_length=100, num_labels=639, name="isco68mul",
                                result_filename='result/isco68_mulbert_results.txt',
                                test_labels_filename='result/isco68_mulbert_test_labels.txt',
                                test_predictions_filename='result/isco68_mulbert_test_predictions.txt')


# Print or analyze the results for multilingual BERT model
for config, result in multilingual_bert_results.items():
    print(f"Configuration: {config}")
    print("Test Accuracy:", result['accuracy'])
    print("Test Precision:", result['precision'])
    print("Test Recall:", result['recall'])
    print("Test F1 Score:", result['f1_score'])
    print("Test Cohen's Kappa:", result['cohen_kappa'])
    print("-----------------------------")
