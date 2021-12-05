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

loss_fn = keras.losses.CategoricalCrossentropy(from_logits=False)
#loss_fn=tf.keras.losses.MeanSquaredError()
#opt = tfa.optimizers.SGDW( weight_decay=0.0005, momentum=0.9, clipnorm=1)
opt = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.999, epsilon=1e-07)
metrica = keras.metrics.CategoricalAccuracy(name='Acc')


saved_model=load_model(r'D:\Loss\ResNet20_SHCUT_128\saved_models\cifar10_ResNet20_SHCUT_128_300.h5')
saved_model.compile(loss=loss_fn,optimizer=opt,metrics=metrica)

def direction(model):
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

def loss(model):
    output=model.predict(x_test)
    output /= tf.reduce_sum(output, axis=1, keepdims=True)
    epsilon = tf.convert_to_tensor(0.3, output.dtype.base_dtype)
    output = tf.clip_by_value(output, epsilon, 1. - epsilon)
    CrossEntropy = - tf.reduce_sum(y_test * tf.math.log(output), axis=1)
    return tf.reduce_mean(CrossEntropy).numpy()


# train_loss, train_acc = saved_model.evaluate(train_dataset, verbose=1)
# val_loss, val_acc = saved_model.evaluate(val_dataset, verbose=1)
# test_loss, test_acc = saved_model.evaluate(test_dataset, verbose=1)
# print('Train_Acc: %.3f, Val_Acc: %.3f, Test_Acc: %.3f' % (train_acc, val_acc, test_acc))
# print('Train_loss: %.3f, Val_loss: %.3f, Test_loss: %.3f' % (train_loss, val_loss, test_loss))

def calulate_loss_landscape(model):
    setup_surface_file()
    with h5py.File("./3d_surface_file_ResNet20_SHCUT_128.h5", 'r+') as f:

        xcoordinates = f['xcoordinates'][:]
        ycoordinates = f['ycoordinates'][:]
        losses = f["test_loss"][:]
        accuracies = f["test_acc"][:]

        inds, coords = get_indices(losses, xcoordinates, ycoordinates)

        for count, ind in enumerate(inds):
            print("ind...%s" % ind)
            coord = coords[count]

            beta = direction(model)
            eta = direction(model)

            current_weights=model.get_weights()
            #print('bw', current_weights[0][0][0][0])
            # layer_name = 'conv2d_20'
            # intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
            # intermediate_output = intermediate_layer_model.predict(test_dataset)
            # print('befo',intermediate_output[0][0][0])


            new_weights=[]
            for weight, b,e in zip(current_weights, beta,eta):
                new_weight = weight + b * coord[0] + e * coord[1]
                new_weights.append(new_weight)

            model.set_weights(new_weights)
            model.compile(loss=loss_fn, optimizer=opt, metrics=metrica)

            # layer_name = 'conv2d_20'
            # intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
            # intermediate_output = intermediate_layer_model.predict(test_dataset)
            #print('after',intermediate_output[0][0][0])

            loss, acc = model.evaluate(test_dataset, verbose=1)


            print(loss, acc)

            losses.ravel()[ind] = loss
            accuracies.ravel()[ind] = acc

            print('Evaluating %d/%d  (%.1f%%)  coord=%s' % (
            ind, len(inds), 100.0 * count / len(inds), str(coord)))

            f["test_loss"][:] = losses
            f["test_acc"][:] = accuracies
            f.flush()


def setup_surface_file():
    xmin, xmax, xnum = -1, 1, 10
    ymin, ymax, ynum = -1, 1, 10

    surface_path = "./3d_surface_file_ResNet20_SHCUT_128.h5"

    if os.path.isfile(surface_path):
        print("%s is already set up" % "3d_surface_file_ResNet20_SHCUT_128.h5")

        return

    with h5py.File(surface_path, 'a') as f:
        print("create new 3d_surface_file_ResNet20_SHCUT_128.h5")

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

#calulate_loss_landscape(saved_model)

# names = [weight.name for layer in saved_model.layers for weight in layer.weights]
# weights = saved_model.get_weights()
#
# for name, weight in zip(names, weights):
#     if weight.ndim==1:
#         print(weight.shape)
#         print(name, weight)







