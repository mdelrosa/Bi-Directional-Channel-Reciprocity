import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard, Callback
import scipy.io as sio 
import numpy as np
import math
import time
from DualNet_mag import *
tf.reset_default_graph()

envir = 'indoor' #'indoor' or 'outdoor'
# fit params
epochs = 1000
# image params
img_height = 32
img_width = 32
img_channels = 1 
img_total = img_height*img_width*img_channels
# network params
residual_num = 2
encoded_dim = 384  # compress rate=1/4->dim.=384, compress rate=1/8->dim.=128, compress rate=1/12->dim.=43
data_format = "channels_last"
def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    step = 500
    lr = 1e-3
    if epoch > step + 180:
        lr *= 1e-3
    elif epoch > step + 160:
        lr *= 1e-2
    elif epoch > step + 120:
        lr *= 5e-2
    elif epoch > step + 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses_train = []
        self.losses_val = []

    def on_batch_end(self, batch, logs={}):
        self.losses_train.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs={}):
        self.losses_val.append(logs.get('val_loss'))

def get_data_shape(T,img_channels,img_height,img_width,data_format):
    if(data_format=="channels_last"):
        shape = (T, img_height, img_width, img_channels)
    elif(data_format=="channels_first"):
        shape = (T, img_channels, img_height, img_width)
    return shape

# build CsiNet
if(data_format == "channels_first"):
    auxiliary_input = Input(shape=(img_channels, img_height, img_width))
elif(data_format == "channels_last"):
    auxiliary_input = Input(shape=(img_height, img_width, img_channels))
autoencoder = DualNet_mag(img_channels, img_height, img_width, encoded_dim, aux=auxiliary_input, data_format=data_format) # CSINet with M_1 dimensional latent space
autoencoder.compile(optimizer='adam', loss='mse')
print(autoencoder.summary())

# Data loading
if envir == 'indoor':
    mat = sio.loadmat('../Bi-Directional-Channel-Reciprocity/data/indoor53/Data100_Htrainin_mag.mat')
    x_train = mat['HD_train']
    x_train_up = mat['HU_train']
    mat = sio.loadmat('../Bi-Directional-Channel-Reciprocity/data/indoor53/Data100_Hvalin_mag.mat')
    x_val = mat['HD_val']
    x_val_up = mat['HU_val']

    x_test = x_val
    x_test_up = x_val_up

elif envir == 'outdoor':
    mat = sio.loadmat('../Bi-Directional-Channel-Reciprocity/data/urban3/Data100_Htrainin_mag.mat')
    x_train = mat['HD_train']
    x_train_up = mat['HU_train']
    mat = sio.loadmat('../Bi-Directional-Channel-Reciprocity/data/urban3/Data100_Hvalin_mag.mat')
    x_val = mat['HD_val']
    x_val_up = mat['HU_val']

    x_test = x_val
    x_test_up = x_val_up

x_train = x_train.astype('float32')
x_train_up = x_train_up.astype('float32')
x_val = x_val.astype('float32')
x_val_up = x_val_up.astype('float32')
x_test = x_test.astype('float32')
x_test_up = x_test_up.astype('float32')

x_train = np.reshape(x_train, get_data_shape(len(x_train), img_channels, img_height, img_width, data_format))
x_train_up = np.reshape(x_train_up, get_data_shape(len(x_train_up), img_channels, img_height, img_width, data_format))
x_val = np.reshape(x_val, get_data_shape(len(x_val), img_channels, img_height, img_width, data_format))
x_val_up = np.reshape(x_val_up, get_data_shape(len(x_val_up), img_channels, img_height, img_width, data_format))
x_test = np.reshape(x_test, get_data_shape(len(x_test), img_channels, img_height, img_width, data_format))
x_test_up = np.reshape(x_test_up, get_data_shape(len(x_test_up), img_channels, img_height, img_width, data_format))

file = 'DualNet_mag_' + (envir) + '_dim' + str(encoded_dim) + time.strftime('_%m_%d')
path = 'result/TensorBoard_%s/1' % file

# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'result/saved_models_Dualnet_mag')
model_name = '%s_model.h5' % file
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

history = LossHistory()

callbacks = [checkpoint, lr_reducer, lr_scheduler, history, TensorBoard(log_dir = path)]

autoencoder.fit([x_train, x_train_up], x_train,
                epochs=700,
                batch_size=200,
                shuffle=True,
                validation_data=([x_val, x_val_up], x_val),
                callbacks=callbacks)


autoencoder.load_weights(filepath)

# Testing data
tStart = time.time()
x_hat = autoencoder.predict([x_test, x_test_up])
tEnd = time.time()
print("It cost %f sec" % ((tEnd - tStart) / x_test.shape[0]))


x_test_mag = np.reshape(x_test[:, 0, :, :], (len(x_test), -1))
x_hat_mag = np.reshape(x_hat[:, 0, :, :], (len(x_hat), -1))

power = np.sum(abs(x_test_mag) ** 2, axis=1)
mse = np.sum(abs(x_test_mag - x_hat_mag) ** 2, axis=1)

# Here is the MSE for magnitude
print("In " + envir + " environment")
print("When dimension is", encoded_dim)
print("MSE is ", 10 * math.log10(np.mean(mse)))
filename = "result/decoded_%s.csv" % file
x_hat1 = np.reshape(x_hat, (len(x_hat), -1))
np.savetxt(filename, x_hat1, delimiter=",")

