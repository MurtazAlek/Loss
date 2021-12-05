from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import h5py
import numpy as np
import pandas as pd

dataset = pd.read_csv("ResNet20_SHCUT_NOWD_128_history_log.csv", delimiter=",")

fig, ax = plt.subplots()
ax1 = ax.twinx()
ax.plot(dataset['epoch'],dataset['loss'], label='train_loss',color='blue',linestyle='solid')
ax.plot(dataset['epoch'],dataset['val_loss'], label='val_loss',color='blue',linestyle='dashed')
ax1.plot(dataset['epoch'],dataset['Acc'], label='train_Acc',color='red',linestyle='solid')
ax1.plot(dataset['epoch'],dataset['val_Acc'], label='val_Acc',color='red',linestyle='dashed')
ax.set_ylabel('Loss')
ax1.set_ylabel('Acc')
ax1.legend(loc=2)
ax.legend(loc=1)
plt.title('ResNet20_SHCUT')
fig.savefig(fname='D:\Loss\ResNet20_SHCUT_WD_256\images\loss_Acc_ResNet20_SHCUT.pdf',
            dpi=300, bbox_inches='tight', format='pdf')

surf_name = "test_loss"

with h5py.File(r'D:\Loss\ResNet20_SHCUT_WD_256\3d_surface_file_ResNet20_SHCUT_128.h5','r') as f:

    Z_LIMIT = 10000000000000
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    x = np.array(f['xcoordinates'][:])
    y = np.array(f['ycoordinates'][:])
    X, Y = np.meshgrid(x, y)
    Z = np.array(f[surf_name][:])
    Z[Z > Z_LIMIT] = Z_LIMIT
    Z = -np.log(Z)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")
    ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True, cmap='summer')
    plt.title('ResNet20_SHCUT_WD_BS128')
    fig.savefig(fname='D:\Loss\ResNet20_SHCUT_WD_256\images\D3_surface_plot_ResNet20_SHCUT.pdf',
                  dpi=300, bbox_inches='tight', format='pdf')



    fig_2 = plt.figure()
    CS = plt.contourf(X, Y, Z, cmap='summer')
    plt.clabel(CS, inline=1, fontsize=8, colors='red')
    plt.title('ResNet20_SHCUT')
    fig_2.savefig(fname='D:\Loss\ResNet20_SHCUT_WD_256\images\D2_contor_plot_ResNet20_SHCUT.pdf',
                  dpi=300, bbox_inches='tight', format='pdf')

plt.show()