#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
# Stage a: Loading and preprocessing the image data
# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# Normalize pixel values to range [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test= x_test.astype('float32') / 255.0
# One-hot encode the target labels.
y_train = to_categorical (y_train, 10)
y_test=to_categorical (y_test, 10)
# Stage b: Defining the model's architecture
model=Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
# Compile the model
model.compile(optimizer='adam',
loss='categorical_crossentropy',
metrics=['accuracy'])
# Stage c: Training the model
history=model.fit(x_train, y_train, epochs=10, batch_size=64,
validation_data=(x_test, y_test))
# Stage d: Estimating the model's performance
test_loss, test_accuracy=model.evaluate(x_test, y_test)
print (f'Test Loss: {test_loss:.4f}')
print (f'Test Accuracy: {test_accuracy: .4f}')


# In[ ]:




