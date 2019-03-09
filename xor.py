# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 09:42:07 2019

@author: Edvan Soares

Exemplo de utilização do MLP com a biblioteca Keras

Exemplo do problema XOR
"""

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np 

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

model = Sequential()
model.add(Dense(2, input_dim=2))
model.add(Activation('tanh'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.1)
model.compile(loss='binary_crossentropy', optimizer=sgd)

model.fit(X, y, batch_size=1, nb_epoch=1000)
print(model.predict_proba(X))