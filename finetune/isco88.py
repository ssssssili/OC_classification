from finetuneBert import train_and_evaluate_series_model
from data_preprocess import CombineFeature,PlotData
import pandas as pd

asial = pd.read_csv('../data/(English - ISCO-88) AL_allcodes(AsiaLymph) - Copy.csv')
asial_prep = asial[~asial['isco88_cg_4'].str.contains('z')]
isco88_data = CombineFeature(asial_prep, column=['occupation_en', 'task_en', 'employer_en', 'product_en'], withname=False)
isco88_data['label'] = isco88_data['isco88_cg_4']
isco88_data = isco88_data[['feature', 'label']]

print(len(isco88_data))
print(len(isco88_data['label'].value_counts()), isco88_data['label'].value_counts().mean(), isco88_data['label'].value_counts().std())
print(max(isco88_data['label'].value_counts()), min(isco88_data['label'].value_counts()))
maxnum = []
for i in isco88_data['feature']:
    cnt = i.split()
    maxnum.append(len(cnt))
print(max(maxnum), min(maxnum))
print(pd.Series(maxnum).mean(), pd.Series(maxnum).std())
PlotData(pd.Series(maxnum))
exit()

layer_configs = [
    []                                 # No layer frozen
    #['classifier', 'pooler']            # Unfreeze only the classifier layer
]

mul = 'bert-base-multilingual-cased'
eng = 'bert-base-cased'

mul_results = train_and_evaluate_series_model(isco88_data['feature'], isco88_data['label'],
                                model_type=mul, layer_configs=layer_configs,
                                batch_size=640, num_epochs=50, max_length=305, num_labels=388, name="isco88mul")

# cased bert
results = train_and_evaluate_series_model(isco88_data['feature'], isco88_data['label'],
                                model_type=eng, layer_configs=layer_configs,
                                batch_size=640, num_epochs=50, max_length=305, num_labels=388, name="isco88eng")




