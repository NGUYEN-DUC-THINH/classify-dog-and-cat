import tensorflow as tf
import keras
from keras.applications import MobileNetV2
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_path ='data/train'
train_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet_v2.preprocess_input).flow_from_directory(
    directory=train_path, target_size=(224,224), batch_size=30)
valid_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet_v2.preprocess_input).flow_from_directory(
    directory=train_path, target_size=(224,224), batch_size=30)


base_model = MobileNetV2(input_shape=(224,224,3),include_top=False,weights='imagenet')
for layer in base_model.layers:
    layer.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(units = 128, activation = 'relu')(x)
x = Dropout(0.5)(x)
x = Dense(units = 2, activation = 'softmax')(x)
model = Model(inputs = base_model.input, outputs = x)
model.compile(loss="binary_crossentropy", optimizer='adam',metrics=["accuracy"])

model.fit(x = train_batches, validation_data = valid_batches, epochs = 30, verbose = 2)
model.save()


