from __future__ import print_function
from keras.models import load_model

from keras import metrics
from keras import losses
import tensorflow as tf
import cifar10 as c
import numpy as np
import h5py
import os
import keras.utils
import logging
from numpy import linalg as LA

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


x_train = x_train.astype('float64') / 255
x_test = x_test.astype('float64') / 255

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

loss_fn = keras.losses.CategoricalCrossentropy(from_logits=False)
#opt = tfa.optimizers.SGDW( weight_decay=0.0005, momentum=0.9, clipnorm=1)
opt = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.999, epsilon=1e-07)
metrica = keras.metrics.CategoricalAccuracy(name='Acc')


saved_model_big=load_model(r'D:\Loss\ResNet20_SHCUT_2000_v2\saved_models\cifar10_ResNet20v1_model.300.h5')
saved_model_small=load_model(r'D:\Loss\ResNet20_SHCUT_128_v2\saved_models\cifar10_ResNet20v1_model.300.h5')
saved_model_big.compile(loss=loss_fn,optimizer=opt,metrics=metrica)
saved_model_small.compile(loss=loss_fn,optimizer=opt,metrics=metrica)


weight_big=saved_model_big.get_weights()
weight_small=saved_model_small.get_weights()
xmin, xmax, xnum = -1, 2, 10
alpha=np.linspace(xmin, xmax, xnum)
def d1plot(weight_big,weight_small,alpha, dataset, y):
    losses = []
    accuracy = []
    for a in alpha:
        print(a)
        new_weights=[]
        for w_b,w_s in zip(weight_big,weight_small):
            w_n=w_s+a*(w_b-w_s)
            print(min(w_n.ravel()), max(w_n.ravel()))
            new_weights.append(w_n)
        saved_model_small.set_weights(new_weights)
        y_pred = saved_model_small.predict(dataset)
        print(y_pred)
        cce = tf.keras.losses.CategoricalCrossentropy()
        loss = cce(y, y_pred).numpy()
        acc = 1 - (np.sum(np.abs(y_pred - y))) / (2 * y.shape[0])
        saved_model_small.set_weights(weight_small)
        losses.append(loss)
        accuracy.append(acc)
    return losses, accuracy
print(d1plot(weight_big,weight_small,alpha,train_dataset,y_train))











