import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


mnist = tf.keras.datasets.mnist # 28x28 images of hand-written digits 0-9 

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

model = tf.keras.models.Sequential() # initialiaz neuro network model 
model.add(tf.keras.layers.Flatten()) # imput layer
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu)) #  hiden layer, contain 128 neurons, 
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu)) #  hiden layer, contain 128 neurons,
model.add(tf.keras.layers.Dense(10, activation = tf.keras.activations.softmax)) #  output layer, contain 10 neurons,

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy']) # configures the model training process
model.fit(x_train, y_train, epochs = 3)


val_loss, val_acc = model.evaluate(x_test, y_test)
print("val_loss = ",val_loss, "val_acc = ",val_acc)

# Model save
model.save('epic_num_reader.keras')
