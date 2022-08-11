import numpy as np
import pandas as pd
from mahakil_improved import MAHAKIL_improved
from imblearn.under_sampling import TomekLinks
import csv
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

def improved_mahakil(trdat):
    print('正在过采样...')
    # extract trdat
    trdat_lab = trdat.Label
    X_resampled = trdat
    trdat = trdat.drop('Label', axis=1)
    mahak = MAHAKIL_improved()
    trdat_len = trdat.shape[0]
    # print(trdat_lab)
    trdat_label = list(trdat_lab.unique())
    trdat_label = list(np.sort(trdat_label, None))
    del trdat_label[0]
    num = 0
    for i in trdat_label:
        num += len(X_resampled[X_resampled['Label'] == i])
    y_resampled = trdat_lab
    n_samples = trdat_len - num  # 多数类数量
    # n_samples = int((trdat_len - 2 * num) / len(trdat_label))
    for i in trdat_label:
        print('当前生成为：', i)
        if (len(X_resampled[X_resampled['Label'] == i]) < n_samples):
            print('当前少数类{}的数量为:{}'.format(i, len(X_resampled[X_resampled['Label'] == i])))
            X = X_resampled[X_resampled['Label'] == i]
            X = X.drop('Label', axis=1)
            X = X.assign(id=range(X.shape[0]))
            X.set_index('id', inplace=True)
            X_new = mahak.fit_sample(X, n_samples - len(X_resampled[X_resampled['Label'] == i]))
            trdat = np.vstack([trdat, X_new])
            y_resampled = np.hstack((y_resampled, np.array([i] * np.sum(len(X_new)))))

    print("过采样完成!")
    return trdat, y_resampled


def TomekLinks_U(data, target):
    print('正在欠采样...')
    tl = TomekLinks(sampling_strategy='all')  # 由于TomekLink 方法无法控制欠采样的数量，故而可将其作为数据清洗的手段，结合其他过采样方法使用·
    x, y = tl.fit_resample(data, target)
    print('欠采样完成！')
    return x, y

def mix_init(data, target):
    features = []
    for i in range(len(data[0])):
        features.append(str(i + 1))
    features.append("Label")
    train_y = target.reshape(target.shape[0], 1)
    train_data = pd.DataFrame(np.concatenate((data, train_y), axis=1), columns=features)
    trdat, trdat_lab = improved_mahakil(train_data)
    trdat_lab = trdat_lab.reshape(trdat_lab.shape[0], 1)
    X_resampled, y_resampled = TomekLinks_U(trdat, trdat_lab)

    return X_resampled,y_resampled

def mahakil_init(data, target):
    features = []
    for i in range(len(data[0])):
        features.append(str(i + 1))
    features.append("Label")
    train_y = target.reshape(target.shape[0], 1)
    train_data = pd.DataFrame(np.concatenate((data, train_y), axis=1), columns=features)
    trdat, trdat_lab = improved_mahakil(train_data)
    trdat_lab = trdat_lab.reshape(trdat_lab.shape[0], 1)

    return trdat, trdat_lab
