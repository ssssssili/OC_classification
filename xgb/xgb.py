import pandas as pd
import data_preprocess
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

lifew = pd.read_csv('../data/(Dutch - ISCO-68) AMIGO_t - Copy.csv', encoding='latin-1')
lifew_prep = lifew[lifew['bjobcode'].astype(str).str.len()==7]
isco68_data = data_preprocess.CombineFeature(lifew_prep, column=['bjobnm','bjobdes','bjobco'], withname= False)
isco68_data['label'] = isco68_data['bjobcode']
isco68_data = isco68_data[['feature', 'label']]

asial = pd.read_csv('../data/(English - ISCO-88) AL_allcodes(AsiaLymph) - Copy.csv')
asial_prep = asial[~asial['isco88_cg_4'].str.contains('z')]
isco88_data = data_preprocess.CombineFeature(asial_prep, column=['occupation_en', 'task_en', 'employer_en', 'product_en'], withname=False)
isco88_data['label'] = isco88_data['isco88_cg_4']
isco88_data = isco88_data[['feature', 'label']]

const = pd.read_csv('../data/(French - PCS-NAF) Constances_CPro_MG_Operas_012021.csv')
const[['code_naf','code_pcs']].dropna(axis=0,how='any')
naf_prep = const[const['code_naf'].str.contains('#') == False]
naf_data = data_preprocess.CombineFeature(naf_prep, column=['profession_txt', 'secteur_txt'], withname= False)
naf_data['label'] = naf_data['code_naf']
naf_data = naf_data[['feature', 'label']]

pcs_prep = const[const['code_pcs'].str.contains('#') == False]
pcs_data = data_preprocess.CombineFeature(pcs_prep, column=['profession_txt', 'secteur_txt'], withname= False)
pcs_data['label'] = pcs_data['code_pcs']
pcs_data = pcs_data[['feature', 'label']]

data_preprocess.XGBModel('bert-base-multilingual-cased', isco68_data, 'isco68-mul-result.txt', 'ISCO68_Mul')
data_preprocess.XGBModel('GroNLP/bert-base-dutch-cased', isco68_data, 'isco68-mul-result.txt', 'ISCO68_Dutch')

data_preprocess.XGBModel('bert-base-multilingual-cased', isco88_data, 'isco88-mul-result.txt', 'ISCO88_Mul')
data_preprocess.XGBModel('bert-base-cased', isco88_data, 'isco88-mul-result.txt', 'ISCO88_Eng')

data_preprocess.XGBModel('bert-base-multilingual-cased', naf_data, 'naf-mul-result.txt', 'NAF_Mul')
data_preprocess.XGBModel('dbmdz/bert-base-french-europeana-cased', naf_data, 'naf-mul-result.txt', 'NAF_Fr')

data_preprocess.XGBModel('bert-base-multilingual-cased', pcs_data, 'pcs-mul-result.txt', 'PCS_Mul')
data_preprocess.XGBModel('dbmdz/bert-base-french-europeana-cased', pcs_data, 'pcs-mul-result.txt', 'PCS_Fr')

"""
results = xg.evals_result()
epochs = len(results['validation_0']['mlogloss'])
x_axis = range(0, epochs)

# xgboost 'mlogloss' plot
fig, ax = plt.subplots(figsize=(9,5))
ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
ax.plot(x_axis, results['validation_1']['mlogloss'], label='Val')
ax.legend()
plt.ylabel('mlogloss')
plt.title('isco68 XGBoost mlogloss', fontdict={'size': 20})
plt.savefig('result/isco68 mlogloss.png')

# xgboost 'merror' plot
fig, ax = plt.subplots(figsize=(9,5))
ax.plot(x_axis, results['validation_0']['merror'], label='Train')
ax.plot(x_axis, results['validation_1']['merror'], label='Val')
ax.legend()
plt.ylabel('merror')
plt.title('isco68 XGBoost merror', fontdict={'size': 20})
plt.savefig('result/isco68 merror.png')
"""




