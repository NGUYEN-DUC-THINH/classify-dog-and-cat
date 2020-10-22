import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np

train_path ='data/train'
valid_path ='data/valid'
batch_size = 64
train_data_gen = ImageDataGenerator(rescale=1./255,
                                    rotation_range=20,
                                    zoom_range=0.15,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.15,
                                    horizontal_flip=True,
                                    fill_mode="nearest")
train_generator = train_data_gen.flow_from_directory(train_path,target_size=(128,128),batch_size = batch_size)
label_map = (train_generator.class_indices)
print(label_map)
valid_data_gen = ImageDataGenerator(rescale=1./255)
train_generator = valid_data_gen.flow_from_directory(valid_path,target_size=(128,128),batch_size = batch_size)

base_model = MobileNetV2(input_shape=(128,128,3),include_top=False,weights='imagenet')
for layer in base_model.layers:
    layer.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(units = 128, activation = 'relu')(x)
x = Dense(units = 64, activation = 'relu')(x)
x = Dense(units = 2, activation = 'softmax')(x)
model = Model(inputs = base_model.input, outputs = x)
model.compile(loss="binary_crossentropy", optimizer = Adam(lr = 1e-4),metrics=["accuracy"])



H = model.fit_generator(train_generator,steps_per_epoch=len(train_generator)/batch_size, validation_data = train_generator,
                    validation_steps= 50, epochs = 30, verbose = 2)

model.save('model/model.mod', save_format="h5")
for i in H.history:
    plt.plot(H.history[i])
    plt.xlabel('epoch')
    plt.ylabel(i)
    plt.savefig('plot/'+ i + '.png')
    plt.clf()
    
    
    
    


