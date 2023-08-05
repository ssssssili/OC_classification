from sklearn.model_selection import train_test_split
from finetuneBert import train_and_evaluate_series_model
import data_preprocess
import pandas as pd


isco88_prep = pd.read_csv('../data/isco88_prep.csv')
isco88_data = data_preprocess.CombineFeature(isco88_prep, column=['occupation_en', 'task_en', 'employer_en', 'product_en'], withname=False)
isco88_data['label'] = isco88_data['isco88_cg_4']
isco88_data = isco88_data[['feature', 'label']]

texts = isco88_data['feature'].tolist()
labels = isco88_data['label'].tolist()

train_texts, temp_texts, train_labels, temp_labels = train_test_split(texts, labels, test_size=0.4, random_state=42)
val_texts, test_texts, val_labels, test_labels = train_test_split(temp_texts, temp_labels, test_size=0.75, random_state=42)

layer_configs = [
    [0],                # Unfreeze only the classifier layer
    [0, 11],            # Unfreeze the classifier and last layer of BERT
    list(range(12)),    # Unfreeze all layers of BERT
    [0, 5, 11]          # Unfreeze the classifier and selected middle layers
]

# Perform training and evaluation for BERT base model
bert_results = train_and_evaluate_series_model(train_texts, train_labels, val_texts, val_labels, test_texts, test_labels,
                                model_type='bert-base-uncased', layer_configs=layer_configs,
                                batch_size=2, num_epochs=50, max_length=305, num_labels=788,
                                result_filename='result/isco88_bert_results.txt',
                                test_labels_filename='result/isco88_bert_test_labels.txt',
                                test_predictions_filename='result/isco88_bert_test_predictions.txt')

# Perform training and evaluation for multilingual BERT model
multilingual_bert_results = train_and_evaluate_series_model(train_texts, train_labels, val_texts, val_labels, test_texts, test_labels,
                                model_type='bert-base-multilingual-uncased', layer_configs=layer_configs,
                                batch_size=2, num_epochs=50, max_length=305, num_labels=788,
                                result_filename='result/isco88_mulbert_results.txt',
                                test_labels_filename='result/isco88_mulbert_test_labels.txt',
                                test_predictions_filename='result/isco88_mulbert_test_predictions.txt')


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
