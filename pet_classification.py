import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_files
import matplotlib.pyplot as plt
from keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten,Dense
from keras.models import Sequential
from keras import optimizers
import random
import os
import re

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

files = "/Users/zhaonan/Desktop/Pet/train"

FAST_RUN = False
IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3

data_gen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        rescale = 1/255)


dataset = data_gen.flow_from_directory("/Users/zhaonan/Desktop/Pet/train/",
                                        target_size = (100,100),
                                        batch_size = 16,
                                        class_mode=  'categorical'
                                        )

print("starting split the dataset")
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

sgd = optimizers.SGD(lr=0.01, clipnorm=1.)
model.compile(loss = 'categorical_crossentropy',optimizer= sgd,metrics=['Accuracy'])
print(model.summary())
model.fit(train_set,epochs = 5, batch_size = 20,verbose = 1)

