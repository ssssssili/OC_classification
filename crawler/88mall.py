import data_preprocess
from transformers import BertModel, BertTokenizer
import pandas as pd
import torch
import numpy as np
from transformers import logging
from torch.optim import Adam
from tqdm import tqdm
from torch import nn
from sklearn.metrics import classification_report
from sklearn import preprocessing


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.labels = [labels[label] for label in df['label']]
        self.texts = [tokenizer(text,
                                padding='max_length',
                                max_length=305,
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
        # self.bert = BertModel.from_pretrained('roberta-base')
        self.bert = BertModel.from_pretrained('bert-base-uncased_layers0_1_2_3_4_5_6_7_8_9_10_11_model.pt')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 388)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer


def train(model, train_data, val_data, learning_rate, epochs):
    train, val = Dataset(train_data), Dataset(val_data)
    # shuffle为Ture训练时打乱样本的结果更好，但是如果自己比较可以选同样的情况不打乱
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)
    # 使用gpu
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    criterion = nn.CrossEntropyLoss()  # 损失函数
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)  # 优化器

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
            | Train Loss: {total_loss_train / len(train_data): .4f} 
            | Train Accuracy: {total_acc_train / len(train_data): .4f} 
            | Val Loss: {total_loss_val / len(val_data): .4f} 
            | Val Accuracy: {total_acc_val / len(val_data): .4f}''')
        Epoch.append(epoch_num + 1)
        Train_loss.append(total_loss_train / len(train_data))
        Train_acc.append(total_acc_train / len(train_data))
        Val_loss.append(total_loss_val / len(val_data))
        Val_acc.append(total_acc_val / len(val_data))

def evaluate(model, test_data):
    test = Dataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=1)
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
            pred_labels.append(output.argmax(dim=1).cpu().numpy()[0])
    print(f'Test Accuracy: {total_acc_test / len(test_data): .4f}')
    print(classification_report(test_labels, pred_labels))

torch.cuda.init()

le = preprocessing.LabelEncoder()
isco88_prep = pd.read_csv('../data/isco88_prep.csv')
isco88_data = data_preprocess.CombineFeature(isco88_prep, column=['occupation_en', 'task_en', 'employer_en', 'product_en'], withname=False)
isco88_data['label'] = le.fit_transform(isco88_data['isco88_cg_4'])
data = isco88_data[['feature', 'label']]

# 去除一个warning的提示
logging.set_verbosity_error()
# 读取预训练模型

unfreeze_0 = ['bert.pooler', 'dropout.', 'linear.', 'relu.']
unfreeze_01 = ['layer.11', 'bert.pooler', 'dropout.', 'linear.', 'relu.']
#unfreeze_015 = ['layer.7', 'layer.11', 'bert.pooler', 'dropout.', 'linear.', 'relu.']

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
# 读取dataframe
# 以下代码设置labels为词典
labels = dict()
col = data.iloc[:, 1]
arrs = col.values
label = set(arrs)
i = 0
for temp in label:
    labels[temp] = i
    i = i + 1
# 以下代码为df添加新的一列，为text几行的集合
np.random.seed(112)
# 按0.8，0.1，0.1划分数据集为训练集，验证集，测试集
df_train, df_val, df_test = np.split(data.sample(frac=1, random_state=42),
                                     [int(.6 * len(data)), int(.7 * len(data))])
# print(len(df_train),len(df_val), len(df_test))
EPOCHS = 10
LR = 2e-5


print('*'*10, 'unfreeze_0', '*'*10)
model1 = BertClassifier()
for name, param in model1.named_parameters():
    param.requires_grad = False
    for ele in unfreeze_0:
        if ele in name:
            param.requires_grad = True
            break
for name, param in model1.named_parameters():
    if param.requires_grad:
        print(name, param.size())
train(model1, df_train, df_val, LR, EPOCHS)
evaluate(model1, df_test)
torch.save(model1.state_dict(), "88m_unfreeze_0_model.pt")


print('*'*10, 'unfreeze_01', '*'*10)
model2 = BertClassifier()
for name, param in model2.named_parameters():
    param.requires_grad = False
    for ele in unfreeze_01:
        if ele in name:
            param.requires_grad = True
            break
for name, param in model2.named_parameters():
    if param.requires_grad:
        print(name, param.size())
train(model2, df_train, df_val, LR, EPOCHS)
evaluate(model2, df_test)
torch.save(model1.state_dict(), "88m_unfreeze_01_model.pt")

"""
print('*'*10, 'unfreeze_all', '*'*10)
model4 = BertClassifier()
train(model4, df_train, df_val, LR, EPOCHS)
evaluate(model4, df_test)
torch.save(model4.state_dict(), "88m_unfreeze_all_model.pt")

print('*'*10, 'unfreeze_015', '*'*10)
model3 = BertClassifier()
for name, param in model3.named_parameters():
    param.requires_grad = False
    for ele in unfreeze_015:
        if ele in name:
            param.requires_grad = True
            break
for name, param in model3.named_parameters():
    if param.requires_grad:
        print(name, param.size())
train(model3, df_train, df_val, LR, EPOCHS)
evaluate(model3, df_test)
torch.save(model3.state_dict(), "88m_unfreeze_015_model.pt")
"""
