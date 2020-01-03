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




test_data = np.load("test_data.npy")/255 #Shape is (200, 168, 308, 3), results will be output to the csv
x_data = np.load("train_data.npy")/255
y_data = np.load("train_labels.npy") #shape is (760,)


x_train, x_validate, y_train, y_validate = train_test_split(x_data,y_data,test_size=0.25,shuffle=True)

num_classes = 7
model = Sequential()
input_shape = (100,100,3)

tf.train.AdamOptimizer(learning_rate=0.01)

#Convolutional layer 1
model.add(Conv2D(32, kernel_size=(10,10),activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(BatchNormalization())

#Convolutional layer 2
model.add(Conv2D(16, kernel_size=(10,10),activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(BatchNormalization())

# #Convolutional layer 3
# model.add(Conv2D(8, kernel_size=(10,10),activation='relu',input_shape=input_shape))
# model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
# model.add(BatchNormalization())

# #Convolutional layer 4
# model.add(Conv2D(4, kernel_size=(10,10),activation='relu',input_shape=input_shape))
# model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
# model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(units=50))
model.add(Dense(units=num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x_train,y_train,
          batch_size=32,
          epochs=5,
          verbose=1,
          validation_data=(x_validate,y_validate)
          )

predicted = model.predict(test_data)
model.save('CNN_4_Layer.neural_net')

#3 layer hits 91% train 87.5% test

