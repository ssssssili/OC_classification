import string
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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
    s = df.value_counts(sort=False).sort_index()
    plt.plot(s.index, s.values)
    plt.xticks()
    plt.show()

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