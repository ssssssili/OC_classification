# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 19:01:25 2023

@author: Xinhao Lan
"""
import data_preprocess
from transformers import BertModel,BertTokenizer
import pandas as pd
from sklearn import preprocessing
import torch
import numpy as np
from transformers import logging
from torch.optim import Adam
from tqdm import tqdm
from torch import nn
import matplotlib.pyplot as plt


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.labels = [labels[label] for label in df['label']]
        self.texts = [tokenizer(text,
                               padding='max_length', 
                               max_length = 512, # 这里bert最多是512，改小不会太影响，只要大于总的句子长度就可以
                               truncation=True,
                               return_tensors="pt")
                      for text in df['feature']]
    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y

class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        #self.bert = BertModel.from_pretrained('roberta-base')
        #self.bert = BertModel.from_pretrained('bert-base-uncased')
        #self.bert = BertModel.from_pretrained('bert-base-cased')
        self.bert = BertModel.from_pretrained('bert-base-multilingual-uncased')
        
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 732)#*******此处的389改为类的数量即可，一定记得需要改变如果切换数据集的话!!!!!!********
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer

def train(model, train_data, val_data, learning_rate, epochs):
    train, val = Dataset(train_data), Dataset(val_data)
    #shuffle为Ture训练时打乱样本的结果更好，但是如果自己比较可以选同样的情况不打乱
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=8, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=8)
    #使用gpu
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    criterion = nn.CrossEntropyLoss() # 损失函数
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)# 优化器

    if use_cuda:
            model = model.cuda()
            criterion = criterion.cuda()
    Train_loss = []
    Train_acc = []
    Val_loss = []
    Val_acc = []
    Epoch = []
    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0
        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.type(torch.LongTensor)
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].type(torch.LongTensor).to(device)
            input_id = train_input['input_ids'].squeeze(1).type(torch.LongTensor).to(device)
            output = model(input_id, mask)
            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
        total_acc_val = 0
        total_loss_val = 0
        with torch.no_grad():
            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)
  
                output = model(input_id, mask)

                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()
                    
                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc
            
        print(
            f'''Epochs: {epoch_num + 1} 
            | Train Loss: {total_loss_train / len(train_data): .3f} 
            | Train Accuracy: {total_acc_train / len(train_data): .3f} 
            | Val Loss: {total_loss_val / len(val_data): .3f} 
            | Val Accuracy: {total_acc_val / len(val_data): .3f}''')
        Epoch.append(epoch_num + 1)
        Train_loss.append(total_loss_train / len(train_data))
        Train_acc.append(total_acc_train / len(train_data))
        Val_loss.append(total_loss_val / len(val_data))
        Val_acc.append(total_acc_val / len(val_data))
    plt.figure(figsize=(12,8), dpi=100)
    plt.plot(Epoch, Train_loss, c='red', label = 'Train')
    plt.plot(Epoch, Val_loss, c='blue', label = 'Val')
    plt.scatter(Epoch, Train_loss, c='red')
    plt.scatter(Epoch, Val_loss, c='blue')
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel("Epochs", fontdict={'size': 16})
    plt.ylabel("Loss", fontdict={'size': 16})
    plt.title("NAF + Plus + Bert + Unfreeze 1 layer", fontdict={'size': 20})
    plt.savefig('exp18/18_loss.png')
    
    plt.figure(figsize=(12,8), dpi=100)
    plt.plot(Epoch, Train_acc, c='red', label = 'Train')
    plt.plot(Epoch, Val_acc, c='blue', label = 'Val')
    plt.scatter(Epoch, Train_acc, c='red')
    plt.scatter(Epoch, Val_acc, c='blue')
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel("Epochs", fontdict={'size': 16})
    plt.ylabel("Accuracy", fontdict={'size': 16})
    plt.title("NAF + Plus + Bert + Unfreeze 1 layer", fontdict={'size': 20})
    plt.savefig('exp18/18_accuracy.png')
              
def evaluate(model, test_data):

    test = Dataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    test_labels = []
    pred_labels = []
    with torch.no_grad():
        for test_input, test_label in test_dataloader:
              test_label = test_label.to(device)
              mask = test_input['attention_mask'].to(device)
              input_id = test_input['input_ids'].squeeze(1).to(device)
              output = model(input_id, mask)
              acc = (output.argmax(dim=1) == test_label).sum().item()
              total_acc_test += acc
              test_labels.append(test_label.cpu().numpy()[0])
              test_labels.append(test_label.cpu().numpy()[1])
              pred_labels.append(output.argmax(dim=1).cpu().numpy()[0])
              pred_labels.append(output.argmax(dim=1).cpu().numpy()[1])
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}') 
    np.savetxt('exp18/test_labels.txt', test_labels, fmt = '%f')
    np.savetxt('exp18/pred_labels.txt', pred_labels, fmt = '%f')    
    
const = pd.read_csv('data/(French - PCS-NAF) Constances_CPro_MG_Operas_012021.csv')

pcs_short = const[~const['code_pcs'].apply(lambda x: type(x) is float or x.find('#') != -1)]
naf_short = const[~const['code_naf'].apply(lambda x: type(x) is float or x.find('#') != -1)]

pcs_prep = pd.read_csv('pcs_prep.csv')
naf_prep = pd.read_csv('naf_prep.csv')

le = preprocessing.LabelEncoder()
pcs_prep['label'] = le.fit_transform(pcs_prep['code_pcs'])
naf_prep['label'] = le.fit_transform(naf_prep['code_naf'])

pcs_feature = data_preprocess.CombineFeature(pcs_prep, column=['numep', 'profession_txt',
                                                                  'secteur_txt'], withname= True)

naf_feature = data_preprocess.CombineFeature(naf_prep, column=['numep', 'profession_txt',
                                                               'secteur_txt'], withname= True)

pcs_data = pcs_feature[['feature', 'label']]
naf_data = naf_feature[['feature', 'label']]

#data = pcs_data #495
data = naf_data #732
#去除一个warning的提示
logging.set_verbosity_error()
#读取预训练模型
BERT_PATH = 'bert-base-multilingual-uncased'
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
#读取dataframe
#data = pd.read_csv('/home/bme001/20225898/ISCO-88.csv', sep = ',')
#以下代码设置labels为词典
labels = dict()
col = data.iloc[:,1]
arrs = col.values
label = set(arrs)
i = 0
for temp in label:
    labels[temp] = i 
    i = i + 1
#以下代码为df添加新的一列，为text几行的集合

np.random.seed(112)
#按0.8，0.1，0.1划分数据集为训练集，验证集，测试集
df_train, df_val, df_test = np.split(data.sample(frac=1, random_state=42), 
                                     [int(.6*len(data)), int(.9*len(data))])
#print(len(df_train),len(df_val), len(df_test))   
EPOCHS = 10
# EPOCHS = 15
model = BertClassifier()
LR = 1e-5
unfreeze_layers = ['layer.11','bert.pooler', 'dropout.', 'linear.', 'relu.']
for name, param in model.named_parameters():
    print(name,param.size())
 
print("*"*30)
print('\n')
 
for name ,param in model.named_parameters():
    param.requires_grad = False
    for ele in unfreeze_layers:
        if ele in name:
            param.requires_grad = True
            break
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name,param.size())
# LR = 1e-4
train(model, df_train, df_val, LR, EPOCHS) 
evaluate(model, df_test)
