from finetuneBert import train_and_evaluate_series_model
from data_preprocess import CombineFeature, PlotData
import pandas as pd

const = pd.read_csv('../data/(French - PCS-NAF) Constances_CPro_MG_Operas_012021.csv')
const[['code_naf','code_pcs']].dropna(axis=0,how='any')
naf_prep = const[const['code_naf'].str.contains('#') == False]
naf_data = CombineFeature(naf_prep, column=['profession_txt', 'secteur_txt'], withname= False)
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
                                batch_size=32, num_epochs=100, max_length=132, num_labels=732, name="nafmul")

results = train_and_evaluate_series_model(naf_data['feature'], naf_data['label'],
                                model_type=fr, layer_configs=layer_configs,
                                batch_size=32, num_epochs=100, max_length=132, num_labels=732, name="naffr")

