from finetuneBert import train_and_evaluate_series_model
import data_preprocess
import pandas as pd


isco88_prep = pd.read_csv('../data/isco88_prep.csv')
isco88_data = data_preprocess.CombineFeature(isco88_prep, column=['occupation_en', 'task_en', 'employer_en', 'product_en'], withname=False)
isco88_data['label'] = isco88_data['isco88_cg_4']
isco88_data = isco88_data[['feature', 'label']]

"""
maxnum = []
for i in isco88_data['feature']:
    cnt = 0
    for j in i:
        cnt += 1
    maxnum.append(cnt)
print(max(maxnum))
print(len(isco88_data['label'].value_counts()))
"""

mul_12 = "bert-base-multilingual-uncased_layers-1_-2_model.pt"
mul_1 = "bert-base-multilingual-uncased_layers-1_model.pt"
mul_all = "bert-base-multilingual-uncased_layers0_1_2_3_4_5_6_7_8_9_10_11_model.pt"
bert_12 = "bert-base-uncased_layers-1_-2_model.pt"
bert_1 = "bert-base-uncased_layers-1_model.pt"
bert_all = "bert-base-uncased_layers0_1_2_3_4_5_6_7_8_9_10_11_model.pt"

layer_configs = [
    [0, 11]            # Unfreeze classifier and last layers of BERT
    #[0, 5, 11]          # Unfreeze the classifier and selected middle layers
]

# Perform training and evaluation for BERT base model
bert_results = train_and_evaluate_series_model(isco88_data['feature'], isco88_data['label'],
                                model_type='bert-base-uncased', pt_path=bert_all, layer_configs=layer_configs,
                                batch_size=2, num_epochs=10, max_length=305, num_labels=388, name="isco88bert",
                                result_filename='result/isco88_bert_results.txt',
                                test_labels_filename='result/isco88_bert_test_labels.txt',
                                test_predictions_filename='result/isco88_bert_test_predictions.txt')

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
multilingual_bert_results = train_and_evaluate_series_model(isco88_data['feature'], isco88_data['label'],
                                model_type='bert-base-multilingual-uncased', pt_path=mul_all, layer_configs=layer_configs,
                                batch_size=2, num_epochs=10, max_length=305, num_labels=388, name="isco88mul",
                                result_filename='result/isco88_mulbert_results.txt',
                                test_labels_filename='result/isco88_mulbert_test_labels.txt',
                                test_predictions_filename='result/isco88_mulbert_test_predictions.txt')


# Print or analyze the results for multilingual BERT model
for config, result in multilingual_bert_results.items():
    print(f"Configuration: {config}")
    print("Test Accuracy:", result['accuracy'])
    print("Test Precision:", result['precision'])
    print("Test Recall:", result['recall'])
    print("Test F1 Score:", result['f1_score'])
    print("Test Cohen's Kappa:", result['cohen_kappa'])
    print("-----------------------------")
