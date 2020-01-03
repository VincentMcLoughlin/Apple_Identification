#!/usr/bin/env python3
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Flatten
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split
import tensorflow as tf
import cv2
import numpy as np
from keras.optimizers import SGD



test_data = np.load("balanced_test_data.npy")/255 #Shape is (200, 168, 308, 3), results will be output to the csv
x_data = np.load("balanced_train_data.npy")/255
y_data = np.load("balanced_train_labels.npy") #shape is (760,)


x_train, x_validate, y_train, y_validate = train_test_split(x_data,y_data,test_size=0.2,shuffle=True)

num_classes = 7
model = Sequential()
input_shape = (100,100,3)

tf.train.AdamOptimizer(learning_rate=0.01)

model.add(Dense(units=20, input_shape=input_shape, activation='relu'))
model.add(Dense(units=15, input_shape=input_shape, activation='relu'))
model.add(Dense(units=15, input_shape=input_shape, activation='relu'))
model.add(Flatten())
model.add(Dense(units=num_classes, activation='softmax'))

#opt = SGD(lr=0.1)
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss = "sparse_categorical_crossentropy", optimizer = sgd, metrics=['accuracy'])
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x_train,y_train,
          batch_size=32,
          epochs=5,
          verbose=1,
          validation_data=(x_validate,y_validate)
          )

predicted = model.predict(test_data)
model.save('ANN.neural_net')

#3 layer hits 91% train 87.5% test

