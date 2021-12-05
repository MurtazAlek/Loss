import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, ZeroPadding2D
from keras.layers import AveragePooling2D, Input, Flatten
import tensorflow_addons as tfa
from keras.callbacks import LearningRateScheduler
from keras import metrics
from keras import losses
from keras.models import Model
import tensorflow as tf
import cifar10 as c
import numpy as np
import os
import keras.utils
import logging

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

batch_size = 2000
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

def lr_schedule(epoch):
    lr = 0.01
    if epoch > 275:
        lr *= 0.001
    elif epoch > 225:
        lr *= 0.01
    elif epoch > 150:
        lr *= 0.1
    return lr

path=r'D:\Loss\ResNet20_NOSHCUT_2000'

save_dir = os.path.join(path, 'saved_models')
model_name = 'cifar10_ResNet20_NOSHCUT_2000_{epoch:03d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)


ls = LearningRateScheduler(lr_schedule)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=filepath,save_weights_only=False,
                                                               monitor='val_accuracy',mode='max',save_best_only=False)
history=tf.keras.callbacks.CSVLogger("ResNet20_NOSHCUT_2000_history_log.csv", separator=",", append=False)
callback=[ls, model_checkpoint_callback, history]

initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)

def basic_blokc_NOshcut(X,F, s):
    # 1 s=2
    X = Conv2D(filters=F, kernel_size=(3, 3), strides=(s, s), padding='same', kernel_initializer=initializer,  use_bias=False)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=F, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=initializer, use_bias=False)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    # 2
    X = Conv2D(filters=F, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=initializer,  use_bias=False)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=F, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=initializer, use_bias=False)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    # 3
    X = Conv2D(filters=F, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=initializer, use_bias=False)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=F, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=initializer, use_bias=False)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    return X

def basic_blokc_shcut(X,F):
    X_shortcut = X
    # 1
    X = Conv2D(filters=F, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer=initializer, use_bias=False)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=F, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=initializer, use_bias=False)(X)
    X = BatchNormalization(axis=3)(X)
    #X = Activation('relu')(X)

    X_shortcut = Conv2D(filters=F, kernel_size=(1, 1), strides=(2, 2), padding='same', kernel_initializer=initializer, use_bias = False)(X_shortcut)
    X_shortcut = BatchNormalization(axis=3)(X_shortcut)
    X = keras.layers.add([X, X_shortcut])
    X = Activation('relu')(X)
    # 2
    X = Conv2D(filters=F, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=initializer, use_bias=False)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=F, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=initializer, use_bias=False)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    # 3
    X = Conv2D(filters=F, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=initializer, use_bias=False)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=F, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=initializer, use_bias=False)(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    return X

def ResNet20_SHCUT(input_shape=(32,32,3), classes=10):
    X_input = Input(shape=input_shape)
    X=Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=initializer, use_bias=False)(X_input)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = basic_blokc_NOshcut(X,F=16, s=1)
    X = basic_blokc_shcut(X,F=32)
    X = basic_blokc_shcut(X,F=64)

    X = AveragePooling2D((2, 2), name='avg_pool')(X)
    X = Flatten()(X)
    X = Dense(classes, activation='softmax',  use_bias=True)(X)

    model = Model(inputs=X_input, outputs=X, name='ResNet20_SHCUT')
    return model

def ResNet20_NOSHCUT(input_shape=(32,32,3), classes=10):
    X_input = Input(shape=input_shape)
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=initializer, use_bias=False)(X_input)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = basic_blokc_NOshcut(X, F=16, s=1)
    X = basic_blokc_NOshcut(X, F=32, s=2)
    X = basic_blokc_NOshcut(X, F=64, s=2)

    X = AveragePooling2D((2, 2), name='avg_pool')(X)
    X = Flatten()(X)
    X = Dense(classes, activation='softmax',  use_bias=True)(X)

    model = Model(inputs=X_input, outputs=X, name='ResNet20_NOSHCUT')
    return model

model=ResNet20_NOSHCUT(input_shape=(32,32,3),classes=10)

tf.keras.utils.plot_model(model, "ResNet20_NOSHCUT_2000.png", show_shapes=True)

loss_fn = keras.losses.CategoricalCrossentropy(from_logits=False)
#opt = tf.keras.optimizers.SGD(momentum=0.9)
#opt = tfa.optimizers.SGDW(weight_decay=0.0005,momentum=0.9)
opt = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.999, epsilon=1e-07)
metrica = keras.metrics.CategoricalAccuracy(name='Acc')

with tf.device('/device:GPU:0'):
    model.compile(loss=loss_fn,optimizer=opt,metrics=metrica)
    history = model.fit(train_dataset, epochs=epochs, callbacks=[callback], shuffle=True, validation_data=val_dataset)



