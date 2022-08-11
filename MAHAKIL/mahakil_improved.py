# Bennin, K. E., Keung, J., Phannachitta, P., Monden, A., & Mensah, S. (2018). Mahakil: Diversity based oversampling approach to alleviate the class imbalance issue in software defect prediction. IEEE Transactions on Software Engineering, 44(6), 534-550.
from __future__ import division

from collections import Counter
import warnings
import numpy as np
import numpy.linalg
import pandas as pd
import math
from sklearn.utils import _safe_indexing
from scipy.spatial.distance import mahalanobis
import scipy as sp


class MAHAKIL_improved():

    # mahalanobis function computations
    def mahalanob(self, xd, meanCol, IC):
        m = []
        for i in range(xd.shape[0]):
            m.append(mahalanobis(xd.iloc[i, :], meanCol, IC) ** 2)
        return (m)

    # generate new data by average same labels
    def gen_child_data(self, xdata, n_samples):
        n_samples_generated = 0
        # 提取特征:' Destination Port', ' Flow Duration'......
        cols = list(xdata.columns.values)
        newdat = xdata.groupby('mahrank').mean().reset_index()
        newdat = newdat[cols]
        uid = xdata.tail(1).iloc[0]['uid'] + 1
        newdat['uid'] = uid
        xdata['curl'] = 0
        newdat['curl'] = 1
        newdat['par1'] = 1
        newdat['par2'] = 2
        gen_data = newdat
        n_samples_generated += len(gen_data)
        xdata = pd.concat([xdata, gen_data])

        while (n_samples_generated < n_samples):
            # select last generated child(ren) data to be merged with its parents
            extdat = xdata.loc[xdata['curl'] == 1]

            # extract all children
            uids = extdat.uid.unique()
            uid = extdat.tail(1).iloc[0]['uid'] + 1
            # create empty gen array
            # gen_data = np.empty((0, X_class.shape[1]))
            xdata['curl'] = 0
            # gen_data['curl'] = 0
            for l in uids:
                subdat = extdat.loc[extdat['uid'] == l]
                # find the parents of the child
                par1 = subdat.tail(1).iloc[0]['par1']
                par2 = subdat.tail(1).iloc[0]['par2']
                mainpars = [par1, par2]
                # select parents and merge with current data and generate new data
                for gp in mainpars:
                    subdat2 = xdata.loc[xdata['uid'] == gp]
                    mdat = pd.concat([subdat2, subdat])
                    newdat = mdat.groupby('mahrank').mean().reset_index()
                    newdat = newdat[cols]
                    newdat['uid'] = uid
                    uid = uid + 1
                    # convert child data to old and make new data current
                    # subdat['curl'] = 0
                    newdat['curl'] = 1
                    newdat['par1'] = subdat.tail(1).iloc[0]['uid']
                    newdat['par2'] = subdat2.tail(1).iloc[0]['uid']
                    # gen_data = np.append(gen_data, [newdat], axis=0)
                    gen_data = pd.concat([gen_data, newdat])
                    xdata = pd.concat([xdata, newdat])
            n_samples_generated += len(gen_data)

        # actual data to return
        acc_dat = xdata.loc[xdata['curl'] == 0]
        lastdat = xdata.loc[xdata['curl'] == 1]
        act = (len(lastdat)) - (len(gen_data) - n_samples)
        remids = lastdat.uid.unique()
        tk = math.ceil(act / len(remids))
        # select same number from last batch of syn data
        rem_data = pd.DataFrame()
        # np.zeros((tk*len(remids), acc_dat.shape[1]))
        for l in remids:
            subdat = lastdat.loc[lastdat['uid'] == l]
            remdat = subdat.iloc[0:tk]
            rem_data = rem_data.append(remdat)

        # final data

        acc_dat = acc_dat.append(rem_data)
        acc_dat = acc_dat.drop(['mahrank', 'mahd', 'uid', 'par1', 'par2', 'curl'], axis=1)
        return acc_dat

    def fit_sample(self, X, n_samples):
        """A support function that returns artificial samples constructed along
        the line connecting nearest neighbours.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Points from which the points will be created.
        y_type : str or int
            The minority target value, just so the function can return the
            target values for the synthetic variables with correct length in
            a clear format.
        n_samples : int
            The number of samples to generate.
        Returns
        -------
        X_new : {ndarray, sparse matrix}, shape (n_samples_new, n_features)
            Synthetically generated samples.
        y_new : ndarray, shape (n_samples_new,)
            Target values for synthetic samples.
        """

        # compute mahalanobis distance
        x = X  # .iloc[:, 1:]
        Sx = x.cov().values
        det = numpy.linalg.det(Sx)
        if det != 0:
            Sx = numpy.linalg.inv(Sx)
            # 协方差矩阵的行列式为零表明存在行或列的相似度极高，需要消除这些存在问题的行或列
        else:
            # save column name
            coname = list(X.columns.values)
            X[coname] = X[coname].applymap(np.int64)
            # drop correlated features
            # Create correlation matrix
            corr_matrix = X.corr().abs()

            # Select upper triangle of correlation matrix
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

            # Find features with correlation greater than 0.95
            to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

            # Drop features
            X.drop(to_drop, axis=1)
            x = X
            Sx = x.cov().values
            Sx = np.linalg.inv(Sx)

        # 少数类每一个特征的数据均值
        mean = x.mean().values
        md = self.mahalanob(x, mean, Sx)
        # add mahd values
        X_class = X.assign(mahd=md)
        # add the ranks of the mahalanob distance value
        # print('X_class','\n',X_class)
        X_class['mahrank'] = X_class['mahd'].rank(ascending=False)
        # sort descending order
        X_class = X_class.sort_values(by=['mahd'], ascending=False)
        # compute median to use for partitioning
        medd = X_class.median()['mahrank']
        # unique identifier
        X_class['uid'] = 1
        # assign labels to partition into (two)
        k = 1
        w = 1
        for i, row in X_class.iterrows():
            X_class.loc[i, 'mahrank'] = k
            # unique identifier
            X_class.loc[i, 'uid'] = w
            k = k + 1
            # math.ceil(medd)/math.ceil(number_minority/C)
            if (X_class.loc[i, 'mahrank'] == medd):
                k = 1
                X_class.loc[i, 'mahrank'] = k
                # unique identifier
                w = w + 1
                X_class.loc[i, 'uid'] = w
                k = k + 1

        # add mom and dad columns no
        X_class['par1'] = 0
        X_class['par2'] = 0
        X_class['curl'] = 0

        # xdata = X_class
        # print('X_class','\n',X_class)
        X_new = self.gen_child_data(X_class, n_samples)
        # print(X_new)
        # y_new = np.array([class_sample] * np.sum(len(X_new)))
        # X_resampled = np.vstack([X_resampled, X_new])
        # y_resampled = np.hstack((y_resampled, y_new))
        return X_new
        # return X_resampled, y_resampled
