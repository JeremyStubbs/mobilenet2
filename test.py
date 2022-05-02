import numpy as np
import tensorflow as tf
import cv2 as cv
from labels import labels

img = cv.imread("ski.jpg")
inp = cv.resize(img, (224, 224))
inp = inp.reshape(1, 224, 224, 3)


model = tf.keras.applications.MobileNetV2()

predictions = model.predict(inp)

print(predictions.shape)
id = np.where(predictions[0] == np.amax(predictions[0]))
print(id[0][0])
print(labels[id[0][0]+1])



