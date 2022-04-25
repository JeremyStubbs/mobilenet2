import numpy as np
import tensorflow as tf
import cv2 as cv

img = cv.imread("ski.jpg")
print(img.shape, img.size)
inp = cv.resize(img, (224, 224))
print(inp.shape)
# cv.imshow("output",inp)
# inp = inp[:, :, [2, 1, 0]] 
# feed_dict = {'input_1': inp}
inp = inp[np.newaxis, ...]
print(inp.shape)
# inp_img = inp.reshape(1, 224, 224, 3)
# print(inp_img.shape)

model = tf.keras.applications.MobileNetV2()

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

predictions = model.predict(inp)

print(predictions.shape)
print(predictions[0][0])
print(np.where(predictions[0] == np.amax(predictions[0])))

