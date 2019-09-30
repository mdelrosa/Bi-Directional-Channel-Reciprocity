import tensorflow as tf
from tensorflow.keras.layers import concatenate, Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard, Callback
from keras import backend as K
from keras import optimizers
import scipy.io as sio
import numpy as np
import math
import time
import os
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


envir = 'indoor'  # 'indoor' or 'outdoor'
# image params
img_height = 32
img_width = 32
img_channels = 1
img_total = img_height * img_width * img_channels
# network params
residual_num = 2
encoded_dim = 384  # compress rate=1/4->dim.=384, compress rate=1/8->dim.=128, compress rate=1/12->dim.=43

def DualNet_mag(img_channels, img_height, img_width, encoded_dim, residual_num=2, aux=None, encoded_in=None, data_format="channels_last"):

    # Build the autoencoder model of DualNet-mag
    def residual_network(x, auxiliary_input, residual_num, encoded_dim):
        def add_common_layers(y):
            y = BatchNormalization()(y)
            y = LeakyReLU()(y)
            return y

        def residual_block_decoded(y):
            shortcut = y
            y = Conv2D(8, kernel_size=(3, 3), padding='same', data_format=data_format)(y)
            y = add_common_layers(y)

            y = Conv2D(16, kernel_size=(3, 3), padding='same', data_format=data_format)(y)
            y = add_common_layers(y)

            y = Conv2D(2, kernel_size=(3, 3), padding='same', data_format=data_format)(y)
            y = BatchNormalization()(y)

            y = add([shortcut, y])
            y = LeakyReLU()(y)

            return y

        x = Conv2D(1, (3, 3), padding='same', data_format=data_format)(x)
        x = add_common_layers(x)

        x = Reshape((img_total,))(x)
        encoded = Dense(encoded_dim, activation='linear')(x)

        x = Dense(img_total, activation='linear')(encoded)
        if auxiliary_input == None:
            reshape_factor = 1           
        else:
            aux = Reshape((img_total,))(auxiliary_input)
            x = concatenate([x, aux])
            reshape_factor = 2
        if(data_format == "channels_first"):
            x = Reshape((reshape_factor * img_channels, img_height, img_width,))(x)
        if(data_format == "channels_last"):
            x = Reshape((img_height, img_width, reshape_factor * img_channels,))(x)
        for i in range(residual_num):
            x = residual_block_decoded(x)

        x = Conv2D(1, (3, 3), activation='sigmoid', padding='same', data_format=data_format)(x)

        return x

    # reshape based on data_format
    if(data_format == "channels_first"):
        image_tensor = Input(shape=(img_channels, img_height, img_width))
        # auxiliary_input = Input(shape=(img_channels, img_height, img_width))
    elif(data_format == "channels_last"):
        image_tensor = Input(shape=(img_height, img_width, img_channels))
        # auxiliary_input = Input(shape=(img_height, img_width, img_channels))
    network_output = residual_network(image_tensor, aux, residual_num, encoded_dim)
    if aux == None:
        # first network will not use any auxiliary input
        autoencoder = Model(inputs=[image_tensor], outputs=[network_output])
    else:
        # subsequent networks will use an auxiliary input
        autoencoder = Model(inputs=[image_tensor, aux], outputs=[network_output])
    return autoencoder
    # autoencoder.compile(optimizer='adam', loss='mse', metrics=['mse'])
    # print(autoencoder.summary())

    