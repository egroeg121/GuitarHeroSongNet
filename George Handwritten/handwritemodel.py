import keras
import numpy as np
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dropout
from keras.optimizers import SGD

height_res = 28
width_res = 28


def getModel():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=[28,28,1], name='conv1'))
    model.add(Conv2D(32, (3, 3), activation='relu', name='conv2'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='pool1'))
    model.add(Dropout(0.25, name='dropout1'))

    model.add(Conv2D(64, (3, 3), activation='relu', name='conv3'))
    model.add(Conv2D(64, (3, 3), activation='relu', name='conv4'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='pool2'))
    model.add(Dropout(0.25, name='droupout2'))

    model.add(Flatten(name='flatten'))
    model.add(Dense(256, activation='relu', name='dense1'))
    model.add(Dropout(0.5, name='dropout3'))
    model.add(Dense(10, activation='softmax', name='dense2'))


    return model