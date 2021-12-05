from __future__ import print_function
from keras.models import load_model
from keras import models
from keras.models import Model
import tensorflow_addons as tfa
from keras import metrics
from keras import losses
import tensorflow as tf
import cifar10 as c
import numpy as np
import h5py
import math
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


saved_model=load_model(r'D:\Loss\ResNet20_SHCUT_2000_v2\saved_models\cifar10_ResNet20v1_model.300.h5')
saved_model.compile(loss=loss_fn,optimizer=opt,metrics=metrica)

def direction(model):
    #np.random.seed(133)
    direction=[]
    weights=model.get_weights()
    random_dir=[np.random.normal(loc=0, scale=1, size=weight.shape) for weight in weights]
    for weight, dir in zip(weights, random_dir):
        if weight.ndim==4:
            assert weight.shape==dir.shape
            form=weight.shape
            weight=weight.reshape(form[0]*form[1]*form[2], -1)
            dir=dir.reshape(form[0]*form[1]*form[2], -1)
            norm_weight=LA.norm(weight,axis=0)
            norm_dir = LA.norm(dir, axis=0)
            koef=norm_weight/(norm_dir+0.0000001)
            normal_dir=dir*koef
            normal_dir=normal_dir.reshape(form[0],form[1],form[2], -1)
            direction.append(normal_dir)
        elif weight.ndim==2:
            assert weight.shape == dir.shape
            norm_weight = LA.norm(weight, axis=0)
            norm_dir = LA.norm(dir, axis=0)
            koef = norm_weight /( norm_dir+0.0000001)
            normal_dir = dir * koef
            direction.append(normal_dir)
        else:
            dir.fill(0)
            #dir=np.copy(weight)
            direction.append(dir)

    return direction

dx = direction(saved_model)
dy = direction(saved_model)
init_weights=saved_model.get_weights()

def calulate_loss_landscape(model,init_weights,dx,dy):
    setup_surface_file()
    with h5py.File("./3d_surface_file_ResNet20_SHCUT_2000.h5", 'r+') as f:
        xcoordinates = f['xcoordinates'][:]
        ycoordinates = f['ycoordinates'][:]
        losses = f["test_loss"][:]
        accuracies = f["test_acc"][:]

        inds, coords = get_indices(losses, xcoordinates, ycoordinates)

        for count, ind in enumerate(inds):
            print("ind...%s" % ind)
            coord = coords[count]

            new_weights=[]
            for w, d0, d1 in zip(init_weights, dx, dy):
                new_w =  w + coord[0] * d0 + coord[1] * d1
                new_weights.append(new_w)
            model.set_weights(new_weights)
            y_pred = model.predict(test_dataset)
            cce = tf.keras.losses.CategoricalCrossentropy()
            loss = cce(y_test, y_pred).numpy()
            acc = 1 - (np.sum(np.abs(y_pred - y_test))) / (2 * y_test.shape[0])
            model.set_weights(init_weights)

            print(loss, acc)

            losses.ravel()[ind] = loss
            accuracies.ravel()[ind] = acc

            print('Evaluating %d/%d  (%.1f%%)  coord=%s' % (
            ind, len(inds), 100.0 * count / len(inds), str(coord)))

            f["test_loss"][:] = losses
            f["test_acc"][:] = accuracies
            f.flush()


def setup_surface_file():
    xmin, xmax, xnum = -1, 1, 30
    ymin, ymax, ynum = -1, 1, 30

    surface_path = "./3d_surface_file_ResNet20_SHCUT_2000.h5"

    if os.path.isfile(surface_path):
        print("%s is already set up" % "3d_surface_file_ResNet20_SHCUT_2000.h5")

        return

    with h5py.File(surface_path, 'a') as f:
        print("create new 3d_surface_file_ResNet20_SHCUT_2000.h5")

        xcoordinates = np.linspace(xmin, xmax, xnum)
        f['xcoordinates'] = xcoordinates

        ycoordinates = np.linspace(ymin, ymax, ynum)
        f['ycoordinates'] = ycoordinates

        shape = (len(xcoordinates), len(ycoordinates))
        losses = -np.ones(shape=shape)
        accuracies = np.ones(shape=shape)

        f["test_loss"] = losses
        f["test_acc"] = accuracies

        return


def get_indices(vals, xcoordinates, ycoordinates):
    inds = np.array(range(vals.size))
    inds = inds[vals.ravel() <= 0]

    xcoord_mesh, ycoord_mesh = np.meshgrid(xcoordinates, ycoordinates)
    s1 = xcoord_mesh.ravel()[inds]
    s2 = ycoord_mesh.ravel()[inds]

    return inds, np.c_[s1, s2]

calulate_loss_landscape(model=saved_model,init_weights=init_weights, dx=dx,dy=dy)








