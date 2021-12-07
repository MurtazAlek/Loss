from __future__ import print_function
from keras.models import load_model
import matplotlib.pyplot as plt
import random
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

batch_size_small = 128
batch_size_big = 2000
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
train_dataset_small = train_dataset.shuffle(buffer_size=1024).batch(batch_size_small)
train_dataset_big = train_dataset.shuffle(buffer_size=1024).batch(batch_size_big)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset_small = test_dataset.shuffle(buffer_size=1024).batch(batch_size_small)
test_dataset_big = test_dataset.shuffle(buffer_size=1024).batch(batch_size_big)

# val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
# val_dataset = val_dataset.shuffle(buffer_size=1024).batch(batch_size)

loss_fn = keras.losses.CategoricalCrossentropy(from_logits=False)
#opt = tfa.optimizers.SGDW( weight_decay=0.0005, momentum=0.9, clipnorm=1)
opt = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.999, epsilon=1e-07)
metrica = keras.metrics.CategoricalAccuracy(name='Acc')


saved_model_small=load_model(r'D:\Loss\Res_Net20_NOSHCUT_128_v2\saved_models\cifar10_ResNet20_NOSHCUT_128_300.h5')
saved_model_big = load_model(r'D:\Loss\ResNet20_NOSHCUT_2000\saved_models\cifar10_ResNet20_NOSHCUT_2000_300.h5')

weight_small=saved_model_small.get_weights()
weight_big = saved_model_big.get_weights()


xmin, xmax, xnum = -1, 1, 31
alphas=np.linspace(xmin, xmax, xnum)

def Loss_BS(model,alphas,weights,test_dataset,train_dataset):
    np.random.seed(123)
    cce = tf.keras.losses.CategoricalCrossentropy()
    weight_init = [np.random.normal(loc=0.0, scale=1.0, size=w.shape) for w in weights]
    Loss_train=[]
    Loss_test=[]
    #Acc_train=[]
    #Acc_test=[]
    for alpha in alphas:

        new_weights = []
        for w_b, w_init in zip(weights,weight_init):
            if w_b.ndim==4:
                w =(1-alpha)*w_b+alpha*w_init
                new_weights.append(w)
            elif w_b.ndim==2:
                w = (1 - alpha) * w_b + alpha * w_init
                new_weights.append(w)
            else:
                w = w_b
                #w = (1 - alpha) * w_s + alpha * w_init
                #w = np.zeros(w_s.shape)
                new_weights.append(w)

        model.set_weights(new_weights)
        y_pred_test = model.predict(test_dataset)
        y_pred_train = model.predict(train_dataset)

        loss_test = cce(y_test, y_pred_test).numpy()
        loss_train = cce(y_train, y_pred_train).numpy()
        #acc_test = 1 - (np.sum(np.abs(y_pred_test - y_test))) / (2 * y_test.shape[0])
        #acc_train = 1 - (np.sum(np.abs(y_pred_train - y_train))) / (2 * y_train.shape[0])
        model.set_weights(weights)
        Loss_train.append(loss_train)
        Loss_test.append(loss_test)
        #Acc_train.append(acc_train)
        #Acc_test.append(acc_test)
    return Loss_train,Loss_test

fig, ax, = plt.subplots()

fig.suptitle('ResNet20_NOSHCUT_SmallBS_vs_BigBS')
# ax.plot(alphas, Loss_BS(alphas=alphas,model=saved_model_small,weights=weight_small,test_dataset=test_dataset_small,train_dataset=train_dataset_small)[0],
#          label='train_loss',color='blue',linestyle='solid')
ax.plot(alphas, Loss_BS(alphas=alphas,model=saved_model_small,weights=weight_small,test_dataset=test_dataset_small,train_dataset=train_dataset_small)[1],
         label='test_loss_128',color='blue',linestyle='dashed')
# ax.plot(alphas, Loss_BS(alphas=alphas,model=saved_model_big,weights=weight_big,test_dataset=test_dataset_big,train_dataset=train_dataset_big)[0],
#         label='train_loss',color='red',linestyle='solid')
ax.plot(alphas, Loss_BS(alphas=alphas,model=saved_model_big,weights=weight_big,test_dataset=test_dataset_big,train_dataset=train_dataset_big)[1],
        label='test_loss_2000',color='red',linestyle='dashed')
ax.set_ylabel('Loss')
ax.legend(loc=2)
fig.savefig(fname='D:\Loss\Big_BS_vs_Small_BS.png', dpi=300, bbox_inches='tight', format='pdf')
plt.show()
















