#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file loads and returds datasets from datasets folder
"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def _load_diabetes():
    data = pd.read_csv('./datasets/diabetes.csv', delimiter=",",
                       header=None, index_col=False)
    sc = StandardScaler()
    data = sc.fit_transform(data)
    data = pd.DataFrame(data)

    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1]

    # X = preprocessing.scale(X)
    # Y = preprocessing.normalize(Y)

    return X, Y

def _load_iris():
    data = pd.read_csv('./datasets/iris.csv', delimiter=",",
                       header=None, index_col=False)
    class_name, index = np.unique(data.iloc[:, -1], return_inverse=True)
    data.iloc[:, -1] = index
    data = data.loc[data[4] != 2]
    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1]

    return X, Y

def _load_salary():
    names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
             'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
             'hours-per-week', 'native-country', 'target']

    data = pd.read_csv('./datasets/uci_adult_salary.csv', delimiter=",", header=None, names=names)

    data = data[data["workclass"] != "?"]
    data = data[data["occupation"] != "?"]
    data = data[data["native-country"] != "?"]

    # Convert categorical fields #
    categorical_col = ['workclass', 'education', 'marital-status', 'occupation',
                       'relationship', 'race', 'sex', 'native-country', 'target']

    for col in categorical_col:
        b, c = np.unique(data[col], return_inverse=True)
        data[col] = c

    feature_list = names[:14]
    X = data.loc[:, feature_list]
    Y = data[['target']]

    return X, Y

def _load_airfoil():
    data = pd.read_csv("./datasets/uci_airfoil_self_noise.csv",
                       delimiter=",", header=0, index_col=0)
    sc = StandardScaler()
    data = sc.fit_transform(data)
    data = pd.DataFrame(data)

    Y = data.iloc[:, -1]
    X = data.iloc[:, :-1]

    return X, Y
