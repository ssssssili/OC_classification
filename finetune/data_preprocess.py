import string
import pandas as pd
import numpy as np
import torch
from transformers import RobertaModel, RobertaTokenizer
from transformers import BertModel, BertTokenizer
from transformers import DistilBertModel, DistilBertTokenizer
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

"""
def remove_rarewords(text, lan):
    return " ".join([word for word in str(text).split() if word not in stopwords.words(lan)])

def stem_words(text, lan):
    return " ".join([SnowballStemmer(language=lan).stem(word) for word in text.split()])

# regular data preprocessing
def PrepData(dataset, column, lan, lower=bool, punc=bool, stop_word=bool, stemming=bool):
    df = dataset.copy()
    for col in column:
        if lower:
            df[col] = df[col].astype(str).str.lower()
        if punc:
            df[col] = df[col].astype(str).str.translate(str.maketrans('', '', string.punctuation))
        if stop_word:
            df[col] = df[col].astype(str).apply(lambda text: remove_rarewords(text, lan))
        if stemming:
            df[col] = df[col].astype(str).apply(lambda text: stem_words(text, lan))
    return df
"""

# combine all the feature together into one sentence
def CombineFeature(dataset, column, withname = bool):
    df = dataset.copy()
    df['feature'] = ' '
    for col in column:
        if withname:
            name = col.lower().translate(str.maketrans('', '', string.punctuation))
            df[col] = name + ' ' + df[col].astype(str)
        df['feature'] = df['feature'] + ' ' + df[col].astype(str)
        df.drop(col, axis=1, inplace=True)
    return df

# plot the data distribution
def PlotData(df):
    s = df.value_counts()
    plt.plot(range(len(s)), s.values)
    plt.xticks([])
    plt.show()

# embed sentence
class EmbeddingModelR:
    def __init__(self, model_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def sentence_embedding(self, sentences):
        tokens = self.tokenizer.batch_encode_plus(
            sentences,
            add_special_tokens=True,
            padding='longest',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = tokens['input_ids'].to(self.device)
        attention_mask = tokens['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            sentence_embeddings = outputs.last_hidden_state.mean(dim=1)

        return sentence_embeddings.cpu().numpy()


class EmbeddingModelB:
    def __init__(self, model_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def sentence_embedding(self, sentences):
        tokens = self.tokenizer.batch_encode_plus(
            sentences,
            add_special_tokens=True,
            padding='longest',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = tokens['input_ids'].to(self.device)
        attention_mask = tokens['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            sentence_embeddings = outputs.last_hidden_state.mean(dim=1)

        return sentence_embeddings.cpu().numpy()


class EmbeddingModelD:
    def __init__(self, model_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def sentence_embedding(self, sentences):
        tokens = self.tokenizer.batch_encode_plus(
            sentences,
            add_special_tokens=True,
            padding='longest',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = tokens['input_ids'].to(self.device)
        attention_mask = tokens['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            sentence_embeddings = outputs.last_hidden_state.mean(dim=1)

        return sentence_embeddings.cpu().numpy()

# split dataset into subsets regarding sample size
def BuildSubset(y, thershold, num):
    s = pd.Series(y).value_counts()
    split = s.index[0]
    cnt = 1
    index = []
    tmp = []
    for val in s.index:
        if cnt < num:
            if s[split]/s[val] < thershold:
                idx = np.where(y == val)[0]
                tmp.extend(list(idx))
            else:
                index.append(tmp)
                tmp = []
                split = val
                cnt += 1
        else:
            idx = np.where(y == val)[0]
            tmp.extend(list(idx))
    index.append(tmp)

    for i in range(num-cnt):
        index.append([])

    return index

# train, test, validation sets split, let training set has all class
def SplitDataset(x, y, training, test):
    tem = []
    x_mul = []
    y_mul = []
    x_sin = []
    y_sin = []
    for l in np.unique(y):
        if pd.Series(y).value_counts()[l] == 1:
            tem.append(l)

    for i in range(len(y)):
        if y[i] in tem:
            x_sin.append(x[i])
            y_sin.append(y[i])
        else:
            x_mul.append(x[i])
            y_mul.append(y[i])

    x_mul = np.array(x_mul)
    y_mul = np.array(y_mul)
    x_sin = np.array(x_sin)
    y_sin = np.array(y_sin)

    if len(np.unique(y_mul)) / len(y_mul) > (1 - training):
        x_train, x_test, y_train, y_test = train_test_split(x_mul, y_mul,
                                                            test_size=(len(np.unique(y_mul)) / len(y_mul)),
                                                            stratify=y_mul, random_state=42)
    else:
        x_train, x_test, y_train, y_test = train_test_split(x_mul, y_mul,
                                                            test_size=(1 - training), stratify=y_mul, random_state=42)

    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test,
                                                    test_size=(test / (1 - training)), random_state=42)

    if len(x_sin) > 0:
        x_train = np.concatenate((x_train, x_sin), axis=0)
        y_train = np.concatenate((y_train, y_sin), axis=0)

    return x_train, x_test, x_val, y_train, y_test, y_val
