from finetuneBert2 import train_and_evaluate_series_model
import data_preprocess
import pandas as pd

lifew = pd.read_csv('../data/(Dutch - ISCO-68) AMIGO_t - Copy.csv', encoding='latin-1')
lifew_prep = lifew[lifew['bjobcode'].astype(str).str.len()==7]
isco68_data = data_preprocess.CombineFeature(lifew_prep, column=['bjobnm'], withname= False)
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
#    [0, 11],            # Unfreeze the classifier and last layer of BERT
#    list(range(12)),    # Unfreeze all layers of BERT
    [0, 7, 11]          # Unfreeze the classifier and selected middle layers
]
multilingual_uncased = 'bert-base-multilingual-uncased'
multilingual_cased = 'bert-base-multilingual-cased'
monolingual_cased = 'GroNLP/bert-base-dutch-cased'

# Preprocess uncased text
isco68_data_un = data_preprocess.PrepData(isco68_data, column=['feature'], lan='dutch', lower=True, punc=True, stop_word=False, stemming=False)

# multilingual uncased bert
mul_un_results = train_and_evaluate_series_model(isco68_data_un['feature'], isco68_data_un['label'],
                                model_type=multilingual_uncased, layer_configs=layer_configs,
                                batch_size=2, num_epochs=100, max_length=100, num_labels=639, name="isco68mulun",
                                result_filename='result/isco68_mulun_results.txt',
                                test_labels_filename='result/isco68_mulun_test_labels.txt',
                                test_predictions_filename='result/isco68_mulun_test_predictions.txt')

# Preprocess cased text
isco68_data_cased = data_preprocess.PrepData(isco68_data, column=['feature'], lan='dutch', lower=False, punc=True, stop_word=False, stemming=False)

# multilingual cased bert
mul_results = train_and_evaluate_series_model(isco68_data_cased['feature'], isco68_data_cased['label'],
                                model_type=multilingual_cased, layer_configs=layer_configs,
                                batch_size=2, num_epochs=100, max_length=100, num_labels=639, name="isco68mul",
                                result_filename='result/isco68_mul_results.txt',
                                test_labels_filename='result/isco68_mul_test_labels.txt',
                                test_predictions_filename='result/isco68_mul_test_predictions.txt')

# cased bert
bert_results = train_and_evaluate_series_model(isco68_data_cased['feature'], isco68_data_cased['label'],
                                model_type=monolingual_cased, layer_configs=layer_configs,
                                batch_size=2, num_epochs=100, max_length=100, num_labels=639, name="isco68",
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
