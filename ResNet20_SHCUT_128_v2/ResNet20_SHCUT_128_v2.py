from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, ZeroPadding2D
from keras.layers import AveragePooling2D, Input, Flatten
import tensorflow_addons as tfa
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras import metrics
from keras import losses
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
import tensorflow as tf
import cifar10 as c
import numpy as np
import os
import keras.utils
import logging

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

batch_size = 128
epochs = 300
num_classes = 10

x_train = c.load_cifar_10_data('cifar-10-batches-py')[0]
y_train = c.load_cifar_10_data('cifar-10-batches-py')[2]
x_test = c.load_cifar_10_data('cifar-10-batches-py')[3]
y_test = c.load_cifar_10_data('cifar-10-batches-py')[5]
label_names = c.load_cifar_10_data('cifar-10-batches-py')[6]


x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

subtract_pixel_mean = True
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean


y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)


inds = np.array(range(x_train.shape[0]))
indexes_val=np.random.choice(inds, 10000, replace=False)
indexes_train=np.setdiff1d(inds,indexes_val)

x_val=x_train[indexes_val]
y_val=y_train[indexes_val]

x_train=x_train[indexes_train]
y_train=y_train[indexes_train]

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.shuffle(buffer_size=1024).batch(batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.shuffle(buffer_size=1024).batch(batch_size)

n = 3
version = 1
depth = n * 6 + 2

model_type = 'ResNet%dv%d' % (depth, version)
input_shape = x_train.shape[1:]

def lr_schedule(epoch):
    lr = 0.01
    if epoch > 275:
        lr *= 0.001
    elif epoch > 225:
        lr *= 0.01
    elif epoch > 150:
        lr *= 0.1
    elif epoch > 100:
        lr *= 0.1
    return lr

def resnet_layer(inputs, num_filters=16,kernel_size=3, strides=1, activation='relu', batch_normalization=True, conv_first=True):
    conv = Conv2D(num_filters, kernel_size=kernel_size, strides=strides,padding='same',
                  kernel_initializer='he_normal',kernel_regularizer=l2(1e-4))
    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def resnet_v1(input_shape, depth, num_classes=10):
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:
                strides = 2
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)
    model = Model(inputs=inputs, outputs=outputs)
    return model
model=resnet_v1(input_shape=input_shape,depth=20,num_classes=10)

tf.keras.utils.plot_model(model, "ResNet20_SHCUT_128.png", show_shapes=True)

loss_fn = keras.losses.CategoricalCrossentropy(from_logits=False)
#opt = tfa.optimizers.SGDW(weight_decay=0.0005,momentum=0.9, nesterov=True, clipnorm=1.0)
opt = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.999, epsilon=1e-07)
metrica = keras.metrics.CategoricalAccuracy(name='Acc')

model.compile(loss=loss_fn,optimizer=opt,metrics=metrica)

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

ls = LearningRateScheduler(lr_schedule)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=filepath,save_weights_only=False,
                                                               monitor='val_accuracy',mode='max',save_best_only=False)
history=tf.keras.callbacks.CSVLogger("ResNet20_SHCUT_128_history_log.csv", separator=",", append=False)
callback=[ls, model_checkpoint_callback, history]

with tf.device('/device:GPU:0'):
    history = model.fit(train_dataset, epochs=epochs, callbacks=[callback], shuffle=True,
                        verbose=1, validation_data=val_dataset)

