import os

import keras
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from tensorflow.keras.applications import *
from PIL import Image
from tensorflow.keras import models
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras import regularizers
lr = 0
IMG_WIDTH = 256
IMG_HEIGHT = 256
COLOR_CHANNELS = 3
NUM_EPOCHS=1
BATCH_SIZE=64
from tensorflow.keras import models


def build_model_gender(learning_rate=lr,fine_tune_at=0):
    model = models.Sequential()
    conv_base = tf.keras.applications.resnet.ResNet152(include_top=False, weights='imagenet',
                                                     input_shape=(IMG_WIDTH, IMG_HEIGHT, COLOR_CHANNELS), classes=2)
    conv_base.trainable = True
    for layer in conv_base.layers[:fine_tune_at]:
        layer.trainable=False
    model.add(conv_base)
    model.add(layers.Flatten())
   #model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1,activation='sigmoid'))
    loss='binary_crossentropy'
    optimizer=optimizers.Adam(learning_rate=lr)
    metric=['accuracy']
    model.compile(loss=loss,optimizer=optimizer,metrics=metric)
    return model

def train_model_gender():
    inp_train_gen = ImageDataGenerator()
    train_data = pd.read_csv('train_images.csv')
    train_data['Gender'] = train_data['Gender'].astype(str)
    train_dataset, val_dataset = train_test_split(train_data, test_size=0.2, stratify=list(train_data['Gender']))
    steps_per_epoch = len(train_data) // BATCH_SIZE
    steps_val_per_epoch = len(val_dataset) // BATCH_SIZE
    train_iterator = inp_train_gen.flow_from_dataframe(train_dataset,
                                                       x_col='image',
                                                       y_col='Gender',
                                                       directory='C2_Demographics2',
                                                       target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                       batch_size=BATCH_SIZE,
                                                       class_mode='binary',
                                                       shuffle=True)
    validation_iterator = inp_train_gen.flow_from_dataframe(val_dataset,
                                                            x_col='image',
                                                            y_col='Gender',
                                                            directory='C2_Demographics2',
                                                            target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                            batch_size=BATCH_SIZE,
                                                            class_mode='binary',
                                                            shuffle=True)
    model = build_model_gender()
    os.makedirs('saved_models',exist_ok=True)
    filepath = "saved_models/transfer_learning_epoch_{epoch:02d}_{val_binary_accuracy:.4f}.h5"
    checkpoint = callbacks.ModelCheckpoint(filepath,
                                           monitor='val_binary_accuracy',
                                           verbose=0,
                                           save_best_only=False)
    callbacks_list = [checkpoint]
    history = model.fit(train_iterator,
                        validation_data=validation_iterator,
                        validation_steps=steps_val_per_epoch,
                        steps_per_epoch=steps_per_epoch,
                        epochs=NUM_EPOCHS,
                        callbacks=callbacks_list)

train_model_gender()