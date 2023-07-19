import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import torch
from transformers import RobertaModel, RobertaTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd


def remove_rarewords(text, lan):
    return " ".join([word for word in str(text).split() if word not in stopwords.words(lan)])

def stem_words(text, lan):
    return " ".join([SnowballStemmer(language=lan).stem(word) for word in text.split()])

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

def NamedFeature(dataset, column):
    df = dataset.copy()
    for col in column:
        name = col.lower().translate(str.maketrans('', '', string.punctuation))
        df[col] = name + ' ' + df[col].astype(str)
    return df

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

def PlotData(dataset, column):
    s = dataset[column].value_counts()
    plt.plot(range(len(s)), s.values)
    plt.show()


class EmbeddingModel:
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


def aggdata(dataset, data, label):
    x = [np.nan] * 768

    for i in np.unique(dataset['label']):
        if i not in np.unique(label):
            data = np.append(data, [np.array(x).reshape(1, 768)])
            label = np.append(label, [i]) 
          
    print(data.shape)
            
    data = data.reshape(-1,768) 
    print(data.shape)
    print(data)

    return data,label

def DataSplit(y, thershold, num):
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


def splitDataset(x, y, training, test):
    x = pd.DataFrame(x)
    y = pd.Series(y)
    tem = []
    x_tem = pd.DataFrame()
    y_tem = pd.Series()
    for l in np.unique(y):
        if y.value_counts()[l] == 1:
            tem.append(l)

    for index, label in y.iteritems():
        if label not in tem:
            x_tem = x_tem.append(x[index])
            y_tem = y_tem.append(pd.Series(label))

    x_train, x_test, y_train, y_test = train_test_split(x_tem, y_tem, test_size=(1 - training),
                                                        stratify=y_tem, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=(test / (1 - training)),
                                                    random_state=42)

    for class_label in tem:
        class_index = np.where(y == class_label)[0][0]
        x_train= x_train.append(pd.Series(x[class_index]))
        y_train = y_train.append(pd.Series(class_label))

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    x_val = np.array(x_val)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_val = np.array(y_val)

    return x_train, x_test, x_val, y_train, y_test, y_val

def ensemble_predict(models, x_test):
    # Perform predictions using each XGBoost model
    predictions = [model.predict(x_test) for model in models]

    # Choose the class with the highest score as the final prediction
    ensemble_predictions = np.argmax(predictions, axis=1)

    return ensemble_predictions