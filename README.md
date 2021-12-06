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






