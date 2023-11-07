# -*- coding: utf-8 -*-
"""Object detection using Transfer Learning of CNN architectures.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1AaYE6xiwqlGn2VD971n7gSRih_cxibDO
"""

# import Required libraries
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# Loading images and labels
(train_ds, train_labels), (test_ds, test_labels) = tfds.load("tf_flowers", split=["train[:70%]", "train[:30%]"], batch_size=-1, as_supervised=True,)  # Include labels

# check existing image size
train_ds[0].shape

# Resizing images
train_ds = tf.image.resize(train_ds, (150, 150))
test_ds = tf.image.resize(test_ds, (150, 150))

# Transforming labels to correct format
train_labels = to_categorical(train_labels, num_classes=5)
test_labels = to_categorical(test_labels, num_classes=5)

#load pretrained model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

# Loading VGG16 model
base_model = VGG16(weights="imagenet", include_top=False, input_shape=train_ds[0].shape)

# freeze parameter
base_model.trainable = False

# Preprocessing input
train_ds = preprocess_input(train_ds)
test_ds = preprocess_input(test_ds)

# model details
base_model.summary()

#add our layers on top of this model
from tensorflow.keras import layers, models

flatten_layer = layers.Flatten()
dense_layer_1 = layers.Dense(50, activation='relu')
dense_layer_2 = layers.Dense(20, activation='relu')
prediction_layer = layers.Dense(5, activation='softmax')

model = models.Sequential([
    base_model,
    flatten_layer,
    dense_layer_1,
    dense_layer_2,
    prediction_layer
])

from tensorflow.keras.callbacks import EarlyStopping

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

es = EarlyStopping(monitor='val_accuracy', mode='max', patience=5,  restore_best_weights=True)

history=model.fit(train_ds, train_labels, epochs=5, validation_split=0.2, batch_size=32, callbacks=[es])

test_loss, test_accuracy = model.evaluate(test_ds,test_labels)
print('Test loss:', test_loss)
print('Test accuracy:', test_accuracy)

