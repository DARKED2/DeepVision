import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

index = 0
loaded_model = tf.keras.models.load_model('epic_num_reader.keras')

mnist = tf.keras.datasets.mnist # 28x28 images of hand-written digits 0-9 

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

predictions = loaded_model.predict(x_test)

print('Predicted label:', np.argmax(predictions[index]))

plt.imshow(x_test[index])
plt.show()

