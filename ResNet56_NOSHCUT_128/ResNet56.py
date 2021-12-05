import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, ZeroPadding2D
from keras.layers import AveragePooling2D, Input, Flatten
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
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

batch_size = 128
epochs = 2
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

# Convert class vectors to binary class matrices.
# y_train = tf.keras.utils.to_categorical(y_train, num_classes)
# y_test = tf.keras.utils.to_categorical(y_test, num_classes)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.shuffle(buffer_size=1024).batch(batch_size)

def basic_blokc_NOshcut(X,F, s):
    # 1 s=2
    X = Conv2D(filters=F, kernel_size=(3, 3), strides=(s, s), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=F, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    # 2
    X = Conv2D(filters=F, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=F, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    # 3
    X = Conv2D(filters=F, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=F, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    # 4
    X = Conv2D(filters=F, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=F, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    # 5
    X = Conv2D(filters=F, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=F, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    # 6
    X = Conv2D(filters=F, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=F, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    # 7
    X = Conv2D(filters=F, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=F, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    # 8
    X = Conv2D(filters=F, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=F, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    # 9
    X = Conv2D(filters=F, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=F, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    return X

def basic_blokc_shcut(X,F):
    X_shortcut = X
    # 1
    X = Conv2D(filters=F, kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=F, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X_shortcut = Conv2D(filters=F, kernel_size=(1, 1), strides=(2, 2), padding='same', use_bias = False)(X_shortcut)
    X_shortcut = BatchNormalization(axis=3)(X_shortcut)
    X = keras.layers.add([X, X_shortcut])
    X = Activation('relu')(X)
    # 2
    X = Conv2D(filters=F, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=F, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    # 3
    X = Conv2D(filters=F, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=F, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    # 4
    X = Conv2D(filters=F, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=F, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    # 5
    X = Conv2D(filters=F, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=F, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    # 6
    X = Conv2D(filters=F, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=F, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    # 7
    X = Conv2D(filters=F, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=F, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    # 8
    X = Conv2D(filters=F, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=F, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    # 9
    X = Conv2D(filters=F, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=F, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    return X

def ResNet56_SHCUT(input_shape=(32,32,3), classes=10):
    X_input = Input(shape=input_shape)
    X=Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X_input)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = basic_blokc_NOshcut(X,F=16, s=1)
    X = basic_blokc_shcut(X,F=32)
    X = basic_blokc_shcut(X,F=64)

    X = AveragePooling2D((2, 2), name='avg_pool')(X)
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', use_bias=True)(X)

    model = Model(inputs=X_input, outputs=X, name='ResNet56_SHCUT')
    return model

def ResNet56_NOSHCUT(input_shape=(32,32,3), classes=10):
    X_input = Input(shape=input_shape)
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X_input)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = basic_blokc_NOshcut(X, F=16, s=1)
    X = basic_blokc_NOshcut(X, F=32, s=2)
    X = basic_blokc_NOshcut(X, F=64, s=2)

    X = AveragePooling2D((2, 2), name='avg_pool')(X)
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', use_bias=True)(X)

    model = Model(inputs=X_input, outputs=X, name='ResNet56_NOSHCUT')
    return model



#model=ResNet56_NOSHCUT(input_shape=(32,32,3),classes=10)


#tf.keras.utils.plot_model(model, "ResNet56_NOSHCUT.png", show_shapes=True)
