#!/usr/bin/python
import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense

import input_data

mnist = input_data.read_data_sets("Mnist_data/", one_hot=True)

X_train = mnist.train.images.reshape(-1, 28, 28)
Y_train = mnist.train.labels

X_test = mnist.test.images.reshape(-1, 28, 28)
Y_test = mnist.test.labels

MODEL_FILENAME = "number_model.hdf5"

# Build the neural network!
model = Sequential()

model.add(Flatten(input_shape=(28,28)))

# Output layer with 32 nodes (one for each possible letter/number we predict)
model.add(Dense(10, activation="softmax"))

# Ask Keras to build the TensorFlow model behind the scenes
#model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the neural network
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=32, epochs=3, verbose=1)

# Save the trained model to disk
model.save(MODEL_FILENAME)
