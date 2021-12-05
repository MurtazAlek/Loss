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

# def myloss(model):
#     output=model.predict(x_test)
#     output /= tf.reduce_sum(output, axis=1, keepdims=True)
#     epsilon = tf.convert_to_tensor(0.1, output.dtype.base_dtype)
#     output = tf.clip_by_value(output, epsilon, 1. - epsilon)
#     CrossEntropy = - tf.reduce_sum(y_test * tf.math.log(output), axis=1)
#     return tf.reduce_mean(CrossEntropy).numpy()

saved_model=load_model(r'D:\Loss\ResNet20_SHCUT_128\saved_models\cifar10_ResNet20_SHCUT_128_300.h5')
saved_model.compile(loss=loss_fn,optimizer=opt,metrics=metrica)

# _, train_acc = saved_model.evaluate(train_dataset, verbose=1)
# _, val_acc = saved_model.evaluate(val_dataset, verbose=1)
# _, test_acc = saved_model.evaluate(test_dataset, verbose=1)
# print('Train_Acc: %.3f, Val_Acc: %.3f, Test_Acc: %.3f' % (train_acc, val_acc, test_acc))

def create_random_directions(model):
    x_direction = create_random_direction(model)
    y_direction = create_random_direction(model)

    return [x_direction, y_direction]


def create_random_direction(model):
    weights = get_weights(model)
    direction = get_random_weights(weights)
    normalize_directions_for_weights(direction, weights)

    return direction


def get_weights(model):
    return [p for p in model.get_weights()]


def get_random_weights(weights):
    return [np.random.normal(loc=0, scale=1, size=w.shape) for w in weights]


def normalize_direction(direction, weights):
    for d, w in zip(direction, weights):
        d*(LA.norm(w) / (LA.norm(d) + 1e-10))


def normalize_directions_for_weights(direction, weights):
    assert (len(direction) == len(weights))
    for d, w in zip(direction, weights):
        normalize_direction(d, w)


def calulate_loss_landscape(model, directions):
    setup_surface_file()
    init_weights = [p for p in model.get_weights()]

    with h5py.File("./3d_surface_file_ResNet20_SHCUT_128.h5", 'r+') as f:

        xcoordinates = f['xcoordinates'][:]
        ycoordinates = f['ycoordinates'][:]
        losses = f["test_loss"][:]
        accuracies = f["test_acc"][:]

        inds, coords = get_indices(losses, xcoordinates, ycoordinates)

        for count, ind in enumerate(inds):
            print("ind...%s" % ind)
            coord = coords[count]
            overwrite_weights(model, init_weights, directions, coord)

            loss, acc = model.evaluate(test_dataset, verbose=1)
            print(loss, acc)

            losses.ravel()[ind] = loss
            accuracies.ravel()[ind] = acc

            print('Evaluating %d/%d  (%.1f%%)  coord=%s' % (
                ind, len(inds), 100.0 * count / len(inds), str(coord)))

            f["test_loss"][:] = losses
            f["test_acc"][:] = accuracies
            f.flush()

            #if ind % 300 == 0:
            #    break

def setup_surface_file():
    xmin, xmax, xnum = -1, 1, 20
    ymin, ymax, ynum = -1, 1, 20

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


def overwrite_weights(model, init_weights, directions, step):
    dx = directions[0]
    dy = directions[1]
    changes = [d0 * step[0] + d1 * step[1] for (d0, d1) in zip(dx, dy)]
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for (p, w, d) in zip(model.get_weights(), init_weights, changes):
        p = w + d

calulate_loss_landscape(saved_model,create_random_directions(saved_model))





