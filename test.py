from tensorflow.keras.preprocessing import image
from tensorflow.keras import models
import cv2
import numpy as np
import argparse
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import mobilenet_v2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
args = vars(ap.parse_args())

model = models.load_model('model/model.mod')

img = cv2.imread(args["image"])
image = cv2.resize(img, (128, 128))
image = (image[...,::-1].astype(np.float32))
image = mobilenet_v2.preprocess_input(image)
image = np.expand_dims(image, axis=0)

a = np.argmax(model.predict(image)[0])
b = np.amax(model.predict(image)[0]) * 100
c ="{:.2f} %".format(b)
category = 'cat' if a == 0 else 'dog'
img = cv2.putText(img, category + ': ' + c , (20,25),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
cv2.imshow('img',img)
cv2.waitKey(0)

