import tensorflow as tf
from tensorflow.keras.layers import concatenate, Lambda, Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU
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
from DualNet_mag import *

envir = 'indoor'  # 'indoor' or 'outdoor'
# image params
img_height = 32
img_width = 32
img_channels = 1
img_total = img_height * img_width * img_channels
# network params
residual_num = 2
# encoded_dim = 384  # compress rate=1/4->dim.=384, compress rate=1/8->dim.=128, compress rate=1/12->dim.=43
M_1 = 384
M_2 = 43
T = 10

def DualNet_mag_temp(img_channels, img_height, img_width, T, M_1, M_2, data_format='channels_last'):
	# base CSINet models
	DualNet_1 = DualNet_mag(img_channels, img_height, img_width, M_1, data_format=data_format) # CSINet with M_1 dimensional latent space
	# plot_model(CsiNet_hi, to_file='CsiNet_hi.png')
	aux = Input((M_1,))
	DualNet_2 = DualNet_mag(img_channels, img_height, img_width, M_2, aux=aux, data_format=data_format) # CSINet with M_2+M_1 dimensional latent space

	print("--- DualNet_mag (no aux input) ---")
	DualNet_1.summary()
	print("--- DualNet_mag (aux input) ---")
	DualNet_2.summary()
	# TO-DO: load weights in hi/lo models
	# load CR=1/4 for M1-generating CsiNet
	# weight_file = get_file(envir, M_1, '09_23')
	# CsiNet_hi.load_weights(weight_file)

	# TO-DO: split large input tensor to use as inputs to 1:T CSINets
	if(data_format == "channels_last"):
		x = Input((T, img_height, img_width, img_channels))
	elif(data_format == "channels_first"):
		x = Input((T, img_channels, img_height, img_width))
	else:
		print("Unexpected data_format param in CsiNet input.") # raise an exception eventually. For now, print a complaint
	# x = Input((T, img_channels, img_height, img_width))
	print('Pre-loop: type(x): {}'.format(type(x)))
	DualNetOut = []
	for i in range(T):
		DualNetIn = Lambda( lambda x: x[:,i,:,:,:])(x)
		print('#{} - type(DualNetIn): {}'.format(i, type(DualNetIn)))
		if i == 0:
			# use CsiNet_hi for t=1
			OutLayer = DualNet_1(DualNetIn)
			# print('#{} - EncodedLayer: {}'.format(i, EncodedLayer))
		else:
			# use CsiNet_lo for t in [2:T]
			# TO-DO: make sure M_1 codeword from CSINet_hi is an aux input to each CSINet_lo
			OutLayer = DualNet_2([OutLayer, DualNetIn])
		print('#{} - OutLayer: {}'.format(i, OutLayer))
		DualNetOut.append(OutLayer)
	
	# TO-DO: apply concatenated CSINet decoder outputs into unrolled LSTM
	# LSTM_model = stacked_LSTM(img_channels, img_height, img_width, T, data_format=data_format)
	# LSTM_model.compile(optimizer='adam', loss='mse')
	# print(LSTM_model.summary())

	# LSTM_in = concatenate(CsiOut)
	# LSTM_out = LSTM_model(LSTM_in)

	DualNetOut = concatenate(DualNetOut)
	if data_format=="channels_last": 
		DualNetOut = Reshape((T,img_height,img_width,img_channels))(DualNetOut)
	elif data_format=="channels_first":
		DualNetOut = Reshape((T,img_channels,img_height,img_width))(DualNetOut)

	# compile full model with large 4D tensor as input and LSTM 4D tensor as output
	full_model = Model(inputs=[x], outputs=[DualNetOut])
	full_model.compile(optimizer='adam', loss='mse')
	full_model.summary()
	return full_model