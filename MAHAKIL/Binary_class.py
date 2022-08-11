import os
from sklearn.model_selection import KFold
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import AllKNN
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
import math
import csv
import MAHAKIL_AND_TOMELINK

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_fscore_support
import warnings
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")

Sampler_name = ['SMOTE', 'TomekLinks', 'MAHAKIL', 'SMOTEENN', 'SMOTETomek', 'MAHAKILTomek']

switch = {'SMOTE': 0,  # 采样函数集合
          'TomekLinks': 1,
          'MAHAKIL': 2,
          'SMOTEENN': 3,
          'SMOTETomek': 4,
          'MAHAKILTomek': 5,
          }
Model = ['BernoulliNB', 'GradientBoostingClassifier', 'KNN']

data_path = 'duo/ant/data.csv'
label_path = 'duo/ant/label.csv'

smote = [[[], [], [], []],
         [[], [], [], []], [[], [], [], []]]

tomelinks = [[[], [], [], []],
             [[], [], [], []], [[], [], [], []]]

somteenn = [[[], [], [], []],
            [[], [], [], []], [[], [], [], []]]

smotetomek = [[[], [], [], []],
              [[], [], [], []], [[], [], [], []]]

mahakil = [[[], [], [], []],
           [[], [], [], []], [[], [], [], []]]

mahaktomek = [[[], [], [], []],
              [[], [], [], []], [[], [], [], []]]

resampled_method = [smote, tomelinks,
                    mahakil,
                    somteenn,
                    smotetomek, mahaktomek]


# 分类指标：precision、recall、f1score、 auc
def save_result(precision, recall, f1score, auc, resampled_seq, model_seq):
    for i in range(4):
        if i == 0:
            resampled_method[resampled_seq][model_seq][i].append(precision)
        if i == 1:
            resampled_method[resampled_seq][model_seq][i].append(recall)
        if i == 2:
            resampled_method[resampled_seq][model_seq][i].append(f1score)
        if i == 3:
            resampled_method[resampled_seq][model_seq][i].append(auc)


def display(Resampled_seq, model_seq, y_true, y_pred, y_pred_proba):
    CM = confusion_matrix(y_true, y_pred)
    tn = CM[0][0]
    fn = CM[1][0]
    tp = CM[1][1]
    fp = CM[0][1]
    # precision
    precision = tp / (tp + fp)
    # recall or sensitivity
    recall = tp / (tp + fn)
    # F1-score
    f1score = metrics.f1_score(y_true, y_pred)
    # AUC
    auc = metrics.roc_auc_score(y_true, y_pred)
    save_result(precision, recall, f1score, auc, Resampled_seq, model_seq)


# pipeline函数就是分类器功能
def pipeline(Resampled_name, X_resampled, y_resampled, tsdat, tsdat_lab):
    # Create_file(Resampled_name)
    # # NB
    # print('NB如下:')
    # # func_name = datsamp + '_NB'
    # clf = GaussianNB()
    # clf.fit(X_resampled, y_resampled)
    # y_pred = clf.predict(tsdat)
    # y_pred_proba = clf.predict_proba(tsdat)
    # display(Resampled_name, 0, tsdat_lab, y_pred, y_pred_proba)
    # print('NB测试完成!')
    # BernoulliNB
    print('BernoulliNB如下:')
    # func_name = datsamp + '_BernoulliNB'
    BNB = BernoulliNB(alpha=1, binarize=0.0)
    BNB.fit(X_resampled, y_resampled)
    bnb_pred = BNB.predict(tsdat)
    bnb_pred_proba = BNB.predict_proba(tsdat)
    display(Resampled_name, 0, tsdat_lab, bnb_pred, bnb_pred_proba)
    print('BernoulliNB测试完成!')
    # LogisticRegressio
    # print('LR如下:')
    # # func_name = datsamp + '_LogisticRegression'
    # LR = LogisticRegression(penalty='l2', max_iter=10000)
    # LR.fit(X_resampled, y_resampled)
    # lr_pred = LR.predict(tsdat)
    # lr_pred_proba = LR.predict_proba(tsdat)
    # display(Resampled_name, 2, tsdat_lab, lr_pred, lr_pred_proba)
    # print('LR测试完成!')
    # GradientBoostingClassifier
    print('GBC如下:')
    # func_name = datsamp + 'GBC'
    GBC = GradientBoostingClassifier(n_estimators=200)
    GBC.fit(X_resampled, y_resampled)
    gbc_pred = GBC.predict(tsdat)
    gbc_pred_proba = GBC.predict_proba(tsdat)
    display(Resampled_name, 1, tsdat_lab, gbc_pred, gbc_pred_proba)
    print('GBC测试完成!')
    # tree.DecisionTreeClassifier
    # print('DTC如下:')
    # # func_name = datsamp + '_DTC'
    # DTC = tree.DecisionTreeClassifier()
    # DTC.fit(X_resampled, y_resampled)
    # dtc_pred = DTC.predict(tsdat)
    # dtc_pred_proba = DTC.predict_proba(tsdat)
    # display(Resampled_name, 4, tsdat_lab, dtc_pred, dtc_pred_proba)
    # RF  Instantiate model with 1000 decision trees
    # print('RF如下:')
    # # func_name = datsamp + '_RF'
    # rf = RandomForestClassifier(n_estimators=1000, random_state=42)
    # rf.fit(X_resampled, y_resampled)
    # rf_pred = rf.predict(tsdat)
    # rf_pred_proba = rf.predict_proba(tsdat)
    # display(Resampled_name, 5, tsdat_lab, rf_pred, rf_pred_proba)
    # print('RF测试完成!')
    # KNN
    print('KNN如下:')
    # func_name = datsamp + '_KNN'
    neighclas = KNeighborsClassifier()
    neighclas.fit(X_resampled, y_resampled)
    kn_pred = neighclas.predict(tsdat)
    kn_pred_proba = neighclas.predict_proba(tsdat)
    display(Resampled_name, 2, tsdat_lab, kn_pred, kn_pred_proba)
    print('KNN测试完成!')
    # NNET
    # print('NNET如下:')
    # # func_name = datsamp + '_NNET'
    # mlp = MLPClassifier(hidden_layer_sizes=(30, 30, 30))
    # mlp.fit(X_resampled, y_resampled)
    # nnet_pred = mlp.predict(tsdat)
    # nnet_pred_proba = mlp.predict_proba(tsdat)
    # display(Resampled_name, 7, tsdat_lab, nnet_pred, nnet_pred_proba)
    # print('NNET测试完成!')


# def ROS_O(data, target):
#     rds = RandomOverSampler(sampling_strategy='auto', random_state=42)
#     x, y = rds.fit_resample(data, target)
#     return x, y


def SMOTE_O(data, target):
    smo = SMOTE(sampling_strategy='auto', random_state=42)
    # print(data.shape)
    x, y = smo.fit_resample(data, target)
    return x, y


# def ADASYN_O(data, target):
#     ada = ADASYN(sampling_strategy='auto', random_state=42)
#     x, y = ada.fit_resample(data, target)
#     return x, y


# def BorderlineSMOTE_O(data, target):  # 使用BorderlineSMOTE-1
#     bls = BorderlineSMOTE(sampling_strategy='auto', random_state=42, kind='borderline-1')
#     x, y = bls.fit_resample(data, target)
#     return x, y
#
#
# def ClusterCentroids_U(data, target):
#     cc = ClusterCentroids(sampling_strategy='auto', random_state=42)
#     x, y = cc.fit_resample(data, target)
#     return x, y
#
#
# def RandomUnderSampler_U(data, target):
#     ros = RandomUnderSampler(sampling_strategy='auto', random_state=42)
#     x, y = ros.fit_resample(data, target)
#     return x, y
#
#
# def NearMiss_U(data, target):
#     nm = NearMiss(sampling_strategy='auto')
#     x, y = nm.fit_resample(data, target)
#     return x, y


def TomekLinks_U(data, target):
    tl = TomekLinks(sampling_strategy='auto')
    x, y = tl.fit_resample(data, target)
    return x, y


# def AllKNN_U(data, target):
#     ak = AllKNN(sampling_strategy='auto')
#     x, y = ak.fit_resample(data, target)
#     return x, y


def SMOTEENN_C(data, target):
    se = SMOTEENN(sampling_strategy='auto', random_state=42)
    x, y = se.fit_resample(data, target)
    return x, y


def SMOTETomek_C(data, target):
    st = SMOTETomek(sampling_strategy='auto', random_state=42)
    x, y = st.fit_resample(data, target)
    return x, y


def MAHAKIL_C(data, target):
    x, y = MAHAKIL_AND_TOMELINK.mahakil_init(data, target)
    return x, y


def MAHAKILTomek_C(data, target):
    print(np.unique(target))
    x, y = MAHAKIL_AND_TOMELINK.mix_init(data, target)
    return x, y


alg = [SMOTE_O, TomekLinks_U, MAHAKIL_C, SMOTEENN_C, SMOTETomek_C, MAHAKILTomek_C]


def Resampling_otherMethod(train_data, train_label, test_data, test_label, method):
    X_resampled, y_resampled = alg[switch.get(method)](train_data, train_label)
    pipeline(switch.get(method), X_resampled, y_resampled, test_data, test_label)


def init(train_data, train_label, test_data, test_label):
    # Resampling_otherMethod(train_data, train_label, test_data, test_label, Sampler_name[-1])
    for i in Sampler_name:  # 采样数据
        Resampling_otherMethod(train_data, train_label, test_data, test_label, i)


if __name__ == '__main__':

    data = np.loadtxt(data_path, dtype=float, delimiter=',')
    target = np.loadtxt(label_path, dtype=int)

    kf = KFold(n_splits=10, shuffle=True)
    for train_index, test_index in kf.split(data):
        train_X, train_y = data[train_index], target[train_index]
        test_X, test_y = data[test_index], target[test_index]
        init(train_X, train_y, test_X, test_y)

    for method_index in range(len(resampled_method)):
        print('正在写入', method_index)
        with open(
                'result/22-8-12/duo/' + 'result_' + Sampler_name[
                    method_index] + '.csv',
                'a',
                newline='') as f:
            w = csv.writer(f)

            for j in range(len(resampled_method[method_index])):
                precision_mean = []
                # precision_std = []
                recall_mean = []
                # recall_std = []
                f1socre_mean = []
                # f1score_std = []
                auc_mean = []
                # auc_std = []

                precision_mean.append(np.mean(resampled_method[method_index][j][0]))
                # precision_std.append(np.std(resampled_method[method_index][j][0]))
                recall_mean.append(np.mean(resampled_method[method_index][j][1]))
                # recall_std.append(np.std(resampled_method[method_index][j][1]))
                f1socre_mean.append(np.mean(resampled_method[method_index][j][2]))
                # f1score_std.append(np.std(resampled_method[method_index][j][2]))
                auc_mean.append(np.mean(resampled_method[method_index][j][3]))
                # auc_std.append(np.std(resampled_method[method_index][j][3]))

                result_precision = ['Precision']
                result_recall = ['Recall']
                result_f1score = ['F1score']
                result_auc = ['Auc']
                for i in range(len(precision_mean)):
                    result_precision.append(str(precision_mean[i])[0:6])
                    result_recall.append(str(recall_mean[i])[0:6])
                    result_f1score.append(str(f1socre_mean[i])[0:6])
                    result_auc.append(str(auc_mean[i])[0:6])

                w.writerow([Model[j], 0, 1])
                w.writerow(['Metrics'])
                w.writerow(result_precision)
                w.writerow(result_recall)
                w.writerow(result_f1score)
                w.writerow(result_auc)
                w.writerow('')
            f.close()
