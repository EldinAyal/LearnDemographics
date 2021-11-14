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
from tensorflow.keras import models


def build_model_gender(learning_rate=lr):
    model = models.Sequential()
    conv_base=tf.keras.applications.
