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

def DataSplit(dataset, column, thershold, num):
    s = dataset[column].value_counts()
    split = s.index[0]
    cnt = 1
    index = []
    tmp = []
    for idx in s.index:
        if cnt < num:
            if s[split]/s[idx] < thershold:
                tmp.append(idx)
            else:
                index.append(tmp)
                tmp = []
                split = idx
                cnt += 1
        else:
            tmp.append(idx)
    index.append(tmp)

    return index


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
    df = pd.DataFrame(list(zip(data, label)))
    for i in np.unique(dataset['label']):
        if i not in np.unique(label):
            df.loc[len(df.index)] = [np.array(x).reshape(1, 768), i]

    print(df[0].shape)
    print(type(df[0]))

    return np.array(df[0], dtype=object), np.array(df[1])

