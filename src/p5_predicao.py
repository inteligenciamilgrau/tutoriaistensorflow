# tutorial para predicao de algarismo escritos a mao usando tensorflow com python

import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
print(model.evaluate(x_test, y_test))

# numero da imagem que se deseja prever
prever = 3054

# desenha imagem com matplotlib
import matplotlib.pyplot as plt
plt.imshow(x_test[prever])

#imprime a predicao feita
import numpy as np
print("Predicao:", np.argmax(model.predict(np.array([x_test[prever]]))))

# desenha com OpenCV
import cv2 ## pip install opencv-python
cv2.imshow("Numero", cv2.resize(x_test[prever],(350,350), interpolation = cv2.INTER_CUBIC))
