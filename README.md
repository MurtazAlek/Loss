### Abstract
Visualizing the Loss Landscape of Neural Nets
### Link
https://arxiv.org/abs/1712.09913
### Explanations
<br />
In the paper the authors present the problem of: How the neural network architecture design, batch size / filter size or by using a shortcut connection such as ResNet
can affect the loss surface, from the viewpoint of visualization of the loss function.
<br />
The main contributions of the work are the following:

+ Presenting a visualization method based on “filter normalization” 

+ Observing that with enough layers, the loss landscape changes from nearly convex to higly chaotic

+ Adding skip connections prevents this change to a chaotic landscape

### The Basics of Loss Function Visualization <br />
First of all the authors start with the assertion that Neural Networks are trained on a corpus of feature vectors, for example images, 
and their corresponding labels __(CIFAR10)__. The training is performed by minimizing the loss, and measuring how well our weights predict a label from a data sample. 
These networks contain many parameters, thus, their loss functions are high-dimensional. <br />

#### The 2D (surface) function takes the form: <br />


![equation](https://latex.codecogs.com/svg.image?f(\alpha&space;,\beta&space;)=L(Q^{*}-\alpha&space;\delta-\beta&space;\eta))


The authors in the study use surface plots of the form described above, using a pair of vector - δ and η, that are sampled from a random Gaussian distribution.
<br />
The authors plotting the loss functions using filter-wise normalized directions. The directions for a network with parameters θ are obtained by:
<br />
+ Produce a random Gaussian direction vector d, whose dimensions are compatible with the weights θ.

+ Normalize the filters in d, so they have same norm of the respective filter in θ.


From the paper we can see that the authors performed the second step by replacing each filter of vector d. First, they obtained the Frobenius norm of θ and d. 
After they divided the vector d by the norm of d, and multiplied it by the norm of θ.
An important factor to take into account, is that filter normalization is applied to both the Convolutional Layers and to the Fully Connected ones.
####  Filter Normalized Plots
There are countur plots the loss function using random filter-normalized directions. Although, there are sharpness differences between the large and small batch minima.

![ResNet20_NOSHCUT_128](https://github.com/MurtazAlek/Loss/blob/master/Res_Net20_NOSHCUT_128_v2/images/d2_contor_plot_ResNet20_NOSHCUT_128.pdf)

![ResNet20_NOSHCUT_2000](https://github.com/MurtazAlek/Loss/blob/master/ResNet20_NOSHCUT_2000/images/d2_contor_plot_ResNet20_NOSHCUT_2000.pdf)

![ResNet20_SHCUT_128](https://github.com/MurtazAlek/Loss/blob/master/ResNet20_SHCUT_128_v2/images/d2_contor_plot_ResNet20_SHCUT_128.pdf)

![ResNet20_SHCUT_2000](https://github.com/MurtazAlek/Loss/blob/master/ResNet20_SHCUT_2000_v2/images/d2_contor_plot_ResNet20_SHCUT_2000.pdf)

Another interesting result is that the ability to find global minimizers to neural loss functions is related to the architecture and to the initial training parameters.

![ResNet56_SHCUT_128](https://github.com/MurtazAlek/Loss/blob/master/ResNet56_SHCUT_128_v2/images/d3_surface_file_ResNet56_SHCUT_128.pdf)
![ResNet56_NOSHCUT_128](https://github.com/MurtazAlek/Loss/blob/master/ResNet56_NOSHCUT_128/images/d3_surface_file_ResNet56_NOSHCUT_128.pdf)

![ResNet20_SHCUT_128](https://github.com/MurtazAlek/Loss/blob/master/ResNet20_SHCUT_128_v2/images/d3_surface_file_ResNet20_SHCUT_128.pdf)
![ResNet20_NOSHCUT_128](https://github.com/MurtazAlek/Loss/blob/master/Res_Net20_NOSHCUT_128_v2/images/d3_surface_file_ResNet20_NOSHCUT_128.pdf)


### Experimental Setup
To understand how the architecture affects on the surface structure, there was trained 3 different networks and plotted the landscape surrounding the minimizers (obtained applying the filter-normalized random direction method).
The trained networks are the following:
+ Standard ResNets: ResNets, with different number of layers, namely 20, 56 and optimized for the CIFAR-10 dataset.
+ No-Skip ResNets: ResNets, without skip connections, the same number of layers 20 and 56.

The networks were trained using:
+ CIFAR-10 dataset
+ Adm optimizers beta_1=0.9, beta_2=0.999, epsilon=1e-07
+ Batch size of 128
+ Batch size of 2000
+ 300 epochs
+ Learning rate of 0.01 and a decay by 10 after epochs 150, 225, and 275
+ kernal initializer for No_Skip ResNets random.normal
+ kernal initializer for Standart ResNets 'he normal'

### Personal impression

It is interesting that the influence of the difference in architecture and parameters is beautifully reflected in the shape of the visualized loss function . I understand that the loss function becomes a non-convex function in a model where learning does not proceed, but in that case, it seems necessary to think about how to make it a convex function for each model 



