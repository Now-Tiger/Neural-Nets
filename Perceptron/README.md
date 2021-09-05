# [Perceptron](https://en.wikipedia.org/wiki/Perceptron) 
The Perceptron is a linear machine learning algorithm for binary classification tasks.

It may be considered one of the first and one of the simplest types of artificial neural networks. It is definitely not “deep” learning but is an important building block.

Like logistic regression, it can quickly learn a linear separation in feature space for two-class classification tasks, although unlike logistic regression, it learns using the stochastic gradient descent optimization algorithm and does not predict calibrated probabilities.

# [Perceptron Algorithm](https://ieeexplore.ieee.org/document/80230?arnumber=80230)
The Perceptron algorithm is a two-class (binary) classification machine learning algorithm.

It is a type of neural network model, perhaps the simplest type of neural network model.

It consists of a single node or neuron that takes a row of data as input and predicts a class label. This is achieved by calculating the weighted sum of the inputs and a bias (set to 1). The weighted sum of the input of the model is called the activation.

__Activation = Weights * Inputs + Bias__

If the activation is above 0.0, the model will output 1.0; otherwise, it will output 0.0


__Predict 1: If Activation > 0.0__

__Predict 0: If Activation <= 0.0__


Model weights are updated with a small proportion of the error each batch, and the proportion is controlled by a hyperparameter called the learning rate, typically set to a small value. This is to ensure learning does not occur too quickly, resulting in a possibly lower skill model, referred to as premature convergence of the optimization (search) procedure for the model weights.

__weights(t + 1) = weights(t) + learning_rate (expectedi – predicted) input_i__

Training is stopped when the error made by the model falls to a low level or no longer improves, or a maximum number of epochs is performed.

The learning rate and number of training epochs are hyperparameters of the algorithm that can be set using heuristics or hyperparameter tuning.
