# Load various imports 
import pandas as pd
import os
import librosa
import numpy as np
import sys
import keras
import json
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split 

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 

from keras.callbacks import ModelCheckpoint 
from datetime import datetime 
from WavFileHelper import WavFileHelper

def buildModel(input_shape):
    # create model 
    model = keras.Sequential()
    # 1st conv layer
    model.add(keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3,3), strides=(2,2), padding='same'))
    model.add(keras.layers.BatchNormalization())
    # 2nd conv layer
    model.add(keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3,3), strides=(2,2), padding='same'))
    model.add(keras.layers.BatchNormalization())
    # 3rd conv layer
    model.add(keras.layers.Conv2D(32, (2,2), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((2,2), strides=(2,2), padding='same'))
    model.add(keras.layers.BatchNormalization())
    # flatten output and feed into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))
    # output layer
    model.add(keras.layers.Dense(3, activation='softmax'))

    return model


# Set the path to the full UrbanSound dataset 
data = None
fulldatasetpath = sys.argv[1]
with open(fulldatasetpath) as f:
    data = json.load(f)

# Convert features and corresponding classification labels into numpy arrays
X = np.array(data["mfcc"])
y = np.array(data["labels"])

# split the dataset 
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(x_train.shape)
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]
print(x_train.shape)
# Construct model 
input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
model = buildModel(input_shape)

# Compile the model
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=['accuracy'])

# Display model architecture summary 
model.summary()

# train the CNN
model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=15)

model.save("currentModel.h5")

# evaluate CNN on the test set
test_error, test_accuracy = model.evaluate(x_test, y_test, verbose=1)

print("Accuracy on test is: {}".format(test_accuracy))