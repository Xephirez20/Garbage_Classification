from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization, GlobalAveragePooling2D
import keras.applications.mobilenet_v2 as mobilenetv2
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense
import tensorflow.keras as keras
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.models import Model

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, LeakyReLU
from tensorflow.keras.layers import Input, Dense, Flatten, GlobalAveragePooling2D, MaxPooling2D

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input 
# import cv2 

# #MODEL 2gerbage_classification_model.h5 INI DENSNET201
# #SUDAH MUNCUL OUTPUT PREDICTNYA GAIS BISA GAISS YEYYY!!! :) PUKUL 00:44 =======================================
# def make_model():
#     ModelDenseNet201 = tf.keras.models.Sequential([
#         tf.keras.applications.DenseNet201(input_shape=(224, 224, 3),
#                                           include_top=False,
#                                           pooling='max',
#                                           weights='imagenet'),
#         # tf.keras.layers.GlobalAveragePooling2D(),
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.Dense(256, activation='ReLU'),
#         tf.keras.layers.Dense(128, activation='ReLU'),
#         tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.Dense(64, activation='ReLU'),
#         # 512 neuron hidden layer
#         tf.keras.layers.Dense(12, activation='softmax')
#     ])
#     return ModelDenseNet201


# ARSITEKTUR DARI MODEL MOBILENET V2============ MODEL fix_garbage_classification_model.H5
# sudah bisa juga yeay MUNCUL PREDICTNYA JUGA :) :) PUKUL 1:54 ==================================
IMAGE_WIDTH = 320    
IMAGE_HEIGHT = 320
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3

categories = {0: 'paper', 1: 'cardboard', 2: 'plastic', 3: 'metal', 4: 'trash', 5: 'battery',
              6: 'shoes', 7: 'clothes', 8: 'green-glass', 9: 'brown-glass', 10: 'white-glass',
              11: 'biological'}


def make_model():
    # Create the base MobileNetV2 model
    base_model = MobileNetV2(include_top=False, input_shape=(320, 320, 3))

    # Freeze the base model layers
    base_model.trainable = False

    # Create the model
    model = Sequential()
    model.add(tf.keras.layers.Lambda(lambda x: x * (1/255.0), input_shape=(320, 320, 3)))
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(12, activation='softmax'))

    return model


