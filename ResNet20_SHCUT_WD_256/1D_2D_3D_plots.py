from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import h5py
import numpy as np
import seaborn as sns
import pandas as pd


dataset = pd.read_csv("ResNet20_NOSHCUT_model_history_log.csv", delimiter=",")

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
plt.title('ResNet20_NOSHCUT_WD_BS128')
fig.savefig(fname='D:\Loss_Visual\ResNet20_NOSHCUT_WD_B128\images\loss_Acc_ResNet20_NOSHCUT.pdf',
            dpi=300, bbox_inches='tight', format='pdf')

surf_name = "test_loss"

with h5py.File(r'D:\Loss_Visual\ResNet20_NOSHCUT_WD_B128\3d_surface_file_ResNet20_NOSHCUT.h5','r') as f:

    #Z_LIMIT = 10

    x = np.array(f['xcoordinates'][:])
    y = np.array(f['ycoordinates'][:])

    X, Y = np.meshgrid(x, y)
    Z = np.array(f[surf_name][:])
    # Z[Z > Z_LIMIT] = Z_LIMIT
    # Z = np.log(Z)  # logscale


    fig_1 = plt.figure()
    ax = Axes3D(fig_1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")
    ax.plot_surface(X, Y, Z, linewidth=0, antialiased=False)
    plt.title('ResNet20_SHCUT_WD_BS128')
    fig_1.savefig(fname='D:\Loss_Visual\ResNet20_NOSHCUT_WD_B128\images\D3_surface_plot_ResNet20_NOSHCUT.pdf',
                  dpi=300, bbox_inches='tight', format='pdf')



    fig_2 = plt.figure()
    CS = plt.contour(X, Y, Z, cmap='summer')
    plt.clabel(CS, inline=1, fontsize=8, colors='red')
    plt.title('ResNet20_SHCUT_WD_BS128')
    fig_2.savefig(fname='D:\Loss_Visual\ResNet20_NOSHCUT_WD_B128\images\D2_contor_plot_ResNet20_NOSHCUT.pdf',
                  dpi=300, bbox_inches='tight', format='pdf')


    fig_3 = plt.figure()
    CS = plt.contourf(X, Y, Z, cmap='summer')
    plt.clabel(CS, inline=1, fontsize=8, colors='red')
    plt.title('ResNet20_SHCUT_WD_BS128')
    fig_3.savefig(fname='D:\Loss_Visual\ResNet20_NOSHCUT_WD_B128\images\D2_contorf_plot_ResNet20_NOSHCUT.pdf',
                  dpi=300, bbox_inches='tight', format='pdf')


    plt.figure()
    sns_plot = sns.heatmap(Z, cmap='viridis', cbar=True, vmin=0.5, vmax=1.2,
                               xticklabels=False, yticklabels=False)
    sns_plot.invert_yaxis()
    sns_plot.get_figure().savefig(fname='D:\Loss_Visual\ResNet20_NOSHCUT_WD_B128\images\HeatMap_ResNet20_NOSHCUT.pdf',
                                  dpi=300, bbox_inches='tight', format='pdf')


plt.show()


