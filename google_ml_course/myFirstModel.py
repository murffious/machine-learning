import tensorflow as tf
import numpy as np
from tensorflow import keras

# Define and compile the neural network
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

# Provide the data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

https://developers.google.com/codelabs/tensorflow-1-helloworld?continue=https%3A%2F%2Fdevelopers.google.com%2Flearn%2Fpathways%2Ftensorflow%3Fhl%3Den%23codelab-https%3A%2F%2Fdevelopers.google.com%2Fcodelabs%2Ftensorflow-1-helloworld&hl=en#2
