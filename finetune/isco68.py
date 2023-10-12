from finetuneBert import train_and_evaluate_series_model
import data_preprocess
import pandas as pd

lifew = pd.read_csv('../data/(Dutch - ISCO-68) AMIGO_t - Copy.csv', encoding='latin-1')
lifew_prep = lifew[lifew['bjobcode'].astype(str).str.len()==7]
isco68_data = data_preprocess.CombineFeature(lifew_prep, column=['bjobnm','bjobdes','bjobco'], withname= False)
isco68_data['label'] = isco68_data['bjobcode']
isco68_data = isco68_data[['feature', 'label']]

"""
print(len(isco68_data))
print(len(isco68_data['label'].value_counts()), isco68_data['label'].value_counts().mean(), isco68_data['label'].value_counts().std())
print(max(isco68_data['label'].value_counts()), min(isco68_data['label'].value_counts()))
maxnum = []
for i in isco68_data['feature']:
    cnt = 0
    for j in i:
        cnt += 1
    maxnum.append(cnt)
print(max(maxnum), min(maxnum))
print(pd.Series(maxnum).mean(), pd.Series(maxnum).std())
data_preprocess.PlotData(isco68_data['label'])
data_preprocess.PlotData(pd.Series(maxnum))
"""

layer_configs = [
    [],                                 # No layer frozen
    ['classifier', 'pooler']            # Unfreeze only the classifier layer
]

mul = 'bert-base-multilingual-cased'
dutch = 'GroNLP/bert-base-dutch-cased'

# multilingual
mul_results = train_and_evaluate_series_model(isco68_data['feature'], isco68_data['label'],
                                model_type=mul, layer_configs=layer_configs,
                                batch_size=16, num_epochs=100, max_length=100, num_labels=639, name="isco68mul",
                                result_filename='result/isco68_mul_results.txt')

# cased bert
results = train_and_evaluate_series_model(isco68_data['feature'], isco68_data['label'],
                                model_type=dutch, layer_configs=layer_configs,
                                batch_size=16, num_epochs=100, max_length=100, num_labels=639, name="isco68dutch",
                                result_filename='result/isco68_dutch_results.txt')


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
