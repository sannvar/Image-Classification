# Image Classification 
Project aims to train a shallow neural network using the Keras package in R to predict a subset of images in the FASHION-MNIST dataset.

## Description 
The project trains neural nets with different number of hidden units (32, 128, 256) and activation functions (ReLu, sigmoid) to find the model with the best accuracy. The project accomplishes this by flattening each of the 28x28 pixel images and converting the pixels to grayscale values (0-1)./ All 784 pixels are fed as input to the NN and the output is a binary array corresponding to the image class. The greatest accuracy is found in the model 256 hidden units and the ReLu activation function (88.3%).

## Dataset
The dataset is called by using the dataset_fashion_mnist(). More inforamtion can be found at https://tensorflow.rstudio.com/reference/keras/dataset_fashion_mnist/
