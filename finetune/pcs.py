from finetuneBert import train_and_evaluate_series_model
from data_preprocess import CombineFeature
import pandas as pd

const = pd.read_csv('../data/(French - PCS-NAF) Constances_CPro_MG_Operas_012021.csv')
const[['code_naf','code_pcs']].dropna(axis=0,how='any')
pcs_prep = const[const['code_pcs'].str.contains('#') == False]
pcs_data = CombineFeature(pcs_prep, column=['profession_txt', 'secteur_txt'], withname= False)
pcs_data['label'] = pcs_data['code_pcs']
pcs_data = pcs_data[['feature', 'label']]

layer_configs = [
    [],                                 # No layer frozen
    ['classifier', 'pooler']            # Unfreeze only the classifier layer
]

mul = 'bert-base-multilingual-cased'
fr = 'dbmdz/bert-base-french-europeana-cased'

mul_results = train_and_evaluate_series_model(pcs_data['feature'], pcs_data['label'],
                                model_type=mul, layer_configs=layer_configs,
                                batch_size=32, num_epochs=50, max_length=132, num_labels=495, name="pcsmul")

bert_results = train_and_evaluate_series_model(pcs_data['feature'], pcs_data['label'],
                                model_type=fr, layer_configs=layer_configs,
                                batch_size=32, num_epochs=50, max_length=132, num_labels=495, name="pcsfr")

