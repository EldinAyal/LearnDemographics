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
from tensorflow.keras import models


def build_model_gender(learning_rate=lr):
    model = models.Sequential()
    conv_base = tf.keras.applications.resnet.ResNet152(include_top=False, weights='imagenet',
                                                     input_shape=(IMG_WIDTH, IMG_HEIGHT, COLOR_CHANNELS), classes=2)
    conv_base.trainable = False
    model.add(conv_base)
    print(model.summary())
