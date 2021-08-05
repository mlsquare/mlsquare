#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file loads and returds datasets from datasets folder
"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def _load_1PL_IrtData(size=0.99):
    data = pd.read_csv('./datasets/sim_irt_100_by_100.csv')

    df_new = data[['question_code', 'user_id', 'correctness']]
    df_cols = df_new.columns
    xtrain, xtest, ytrain, ytest = train_test_split(df_new[df_cols[:-1]], df_new[df_cols[-1]], test_size=size)
    x_train_user = pd.get_dummies(xtrain['user_id']).values
    x_train_questions = pd.get_dummies(xtrain['question_code']).values
    y_train= ytrain.values
    return xtrain, y_train, x_train_user, x_train_questions


def _load_simIrt():
    col_name = ['question_code', 'user_id', 'difficulty', 'ability', 'response', 'correctness']
    data = pd.read_csv('./datasets/sim_irt_100_by_100.csv')
    lbe = LabelEncoder()

    data[[col_name[1]]] = data[[col_name[1]]].apply(lambda x: lbe.fit_transform(x))
    data[[col_name[0]]] = data[[col_name[0]]].apply(lambda x: lbe.fit_transform(x))
    X= data.iloc[:,:-1]
    y = data[['correctness']].values
    return X, y

def _load_boston():
    data = pd.read_csv('./datasets/boston.csv')
    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1]

    lbe= LabelEncoder()
    X = X.apply(lambda x: lbe.fit_transform(x))
    return X, Y

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
    _, index = np.unique(data.iloc[:, -1], return_inverse=True)
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
        _, c = np.unique(data[col], return_inverse=True)
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
