from keras.models import load_model
import tensorflow_addons as tfa
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
np.random.seed(123)
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
#opt = tfa.optimizers.SGDW( weight_decay=0.0005, momentum=0.9, clipnorm=1)
opt = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.999, epsilon=1e-07)
metrica = keras.metrics.CategoricalAccuracy(name='Acc')

def myloss(model):
    output=model.predict(x_test)
    output /= tf.reduce_sum(output, axis=1, keepdims=True)
    epsilon = tf.convert_to_tensor(0.1, output.dtype.base_dtype)
    output = tf.clip_by_value(output, epsilon, 1. - epsilon)
    CrossEntropy = - tf.reduce_sum(y_test * tf.math.log(output), axis=1)
    return tf.reduce_mean(CrossEntropy).numpy()

saved_model=load_model(r'D:\Loss\ResNet56_NOSHCUT_128\saved_models\cifar10_ResNet56_NOSHCUT_128_300.h5')
saved_model.compile(loss=loss_fn,optimizer=opt,metrics=metrica)

# _, train_acc = saved_model.evaluate(train_dataset, verbose=1)
# _, val_acc = saved_model.evaluate(val_dataset, verbose=1)
#val_loss , test_acc = saved_model.evaluate(test_dataset, verbose=1)
# print('Train_Acc: %.3f, Val_Acc: %.3f, Test_Acc: %.3f' % (train_acc, val_acc, test_acc))


def calulate_loss_landscape(model, directions):
    setup_surface_file()
    with h5py.File("./3d_surface_file_ResNet56_NOSHCUT_128.h5", 'r+') as f:

        xcoordinates = f['xcoordinates'][:]
        ycoordinates = f['ycoordinates'][:]
        losses = f["test_loss"][:]
        accuracies = f["test_acc"][:]

        inds, coords = get_indices(losses, xcoordinates, ycoordinates)

        for count, ind in enumerate(inds):
            print("ind...%s" % ind)

            coord = coords[count]
            print('wb',model.get_weights()[0][0][0][0])
            model.set_weights(overwrite_weights(model, directions, coord))
            print('wa', model.get_weights()[0][0][0][0])
            #model.compile(loss=loss_fn, optimizer=opt, metrics=metrica)
            #loss, acc = model.evaluate(test_dataset, verbose=1)

            loss = myloss(model)
            acc=1
            #_, acc = model.evaluate(test_dataset, verbose=1)
            print(loss, acc)
            losses.ravel()[ind] = loss
            accuracies.ravel()[ind] = acc

            print('Evaluating %d/%d  (%.1f%%)  coord=%s' % (ind+1, len(inds), 100.0 * (count+1) / len(inds), str(coord)))

            f["test_loss"][:] = losses
            f["test_acc"][:] = accuracies
            f.flush()

def setup_surface_file():
    # xmin, xmax, xnum = -0.000005, 0.000005, 15
    # ymin, ymax, ynum = -0.000005, 0.000005, 15

    xmin, xmax, xnum = -1, 1, 10
    ymin, ymax, ynum = -1, 1, 10


    surface_path = "./3d_surface_file_ResNet56_NOSHCUT_128.h5"

    if os.path.isfile(surface_path):
        print("%s is already set up" % "3d_surface_file_ResNet56_NOSHCUT_128.h5")

        return

    with h5py.File(surface_path, 'a') as f:
        print("create new 3d_surface_file_ResNet56_NOSHCUT_128.h5")

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


def overwrite_weights(model,directions,step):
    new_weights=[]
    dx = directions[0]
    dy = directions[1]
    changes = [d0 * step[0] + d1 * step[1] for (d0, d1) in zip(dx, dy)]
    print('c',changes[0][0][0][0])
    for (w, c) in zip( model.get_weights(), changes):
        new_weights_w = w + c
        new_weights.append(new_weights_w)
    return new_weights

def get_random_weights(weights):
    direction=[]
    for weight in weights:
        form=np.random.normal(loc=0, scale=1, size=weight.shape)
        direction.append(form)
    return direction

def direction_2(model):
    weights=model.get_weights()
    dir=get_random_weights(weights)
    norm_dir=[]
    assert (len(dir) == len(weights))
    for d,w in zip(dir,weights):

        assert (d.shape == w.shape)
        if d.ndim == 1:
            d.fill(0)
            norm_dir.append(np.array(d))
        else:
            norm_dir_d=[]
            for dd,ww in zip(d,w):
                assert (dd.shape == ww.shape)
                dd=dd*(LA.norm(ww)/(LA.norm(dd)+0.00000001))
                norm_dir_d.append(dd)
            norm_dir.append(np.array(norm_dir_d))
    return norm_dir

def directions(model):
    dx=direction_2(model)
    dy=direction_2(model)
    return dx, dy

calulate_loss_landscape(saved_model,directions(saved_model))







def min_max(mass):
    maxes_dx=[]
    mimim_dx=[]
    meann_dx=[]
    for dx in mass:
        maxes_dx.append(max(dx.ravel()))
        mimim_dx.append(min(dx.ravel()))
        meann_dx.append(np.mean(dx.ravel()))
    max_dx=max(maxes_dx)
    min_dx=min(mimim_dx)
    mean_dx = np.mean(meann_dx)
    return min_dx, max_dx, mean_dx

# print(min_max(saved_model.get_weights()))


