# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 09:55:59 2019

@author: Edvan Soares

Exemplo de utilização do MLP com a biblioteca Scikit Learn
"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return str(round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100, 2))

df = pd.read_csv('dataset.csv', sep=';')
train, test = df.iloc[0:len(df.index)-6,:], df.iloc[len(df.index)-6:,:]
x_train = train.iloc[:, 0:-1].values
y_train = train.iloc[:, -1].values
x_test= test.iloc[:, 0:-1].values
y_test = test.iloc[:,-1].values

#MLP
mlp = MLPRegressor(activation="identity", solver="adam", alpha=0.01, hidden_layer_sizes=(200,))
mlp.fit(x_train, y_train)
predicted = mlp.predict(x_test)

plt.title('MAPE='+str(mape(y_test, predicted))+'%)')
plt.plot(y_test, color='b', label='real')
plt.plot(predicted, color='r', label='predicted')
plt.legend()