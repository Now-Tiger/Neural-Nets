# Deep Learning & Neural Networks
![image](https://images.deepai.org/glossary-terms/perceptron-6168423.jpg)

## A neural network :
A neural network is a network or circuit of neurons, or in a modern sense, an artificial neural network, composed of artificial neurons or nodes. Thus a neural network is either a biological neural network, made up of biological neurons, or an artificial neural network, for solving artificial intelligence (AI) problems. The connections of the biological neuron are modeled in artificial neural networks as weights between nodes. A positive weight reflects an excitatory connection, while negative values mean inhibitory connections. All inputs are modified by a weight and summed. This activity is referred to as a linear combination. Finally, an activation function controls the amplitude of the output. For example, an acceptable range of output is usually between 0 and 1, or it could be −1 and 1.

These artificial networks may be used for predictive modeling, adaptive control and applications where they can be trained via a dataset. Self-learning resulting from experience can occur within networks, which can derive conclusions from a complex and seemingly unrelated set of information.

## Artificial intelligence :
A neural network (NN), in the case of artificial neurons called artificial neural network (ANN) or simulated neural network (SNN), is an interconnected group of natural or artificial neurons that uses a mathematical or computational model for information processing based on a connectionistic approach to computation. In most cases an ANN is an adaptive system that changes its structure based on external or internal information that flows through the network.

In more practical terms neural networks are non-linear statistical data modeling or decision making tools. They can be used to model complex relationships between inputs and outputs or to find patterns in data.

The connections are in neural nets are called edges. Neurons and edges typically have a __weight__ that adjusts as learning proceeds. The weight increases or decreases the strength of the signal at a connection. Neurons may have a threshold such that a signal is sent only if the aggregate signal crosses that threshold. Typically, neurons are aggregated into layers. Different layers may perform different transformations on their inputs. Signals travel from the first layer (the input layer), to the last layer (the output layer), possibly after traversing the layers multiple times.

## Types of networks : 
- Artificial Neural Networks (ANN)
- Convolution Neural Networks (CNN)
- Recurrent Neural Networks (RNN)

### Artificial Neural Network (ANN) – What is a ANN and why should you use it?
A single perceptron (or neuron) can be imagined as a Logistic Regression. Artificial Neural Network, or ANN, is a group of multiple perceptrons/ neurons at each layer. ANN is also known as a Feed-Forward Neural network because inputs are processed only in the forward direction

ANN may consists of 3 layers – Input, Hidden and Output. The input layer accepts the inputs, the hidden layer processes the inputs, and the output layer produces the result. Essentially, each layer tries to learn certain weights.
 
ANN can be used to solve problems related to :
- __Tabular data__
- __Image data__
- __Text data__
 
Artificial Neural Network is capable of learning any nonlinear function. Hence, these networks are popularly known as Universal Function Approximators. ANNs have the capacity to learn weights that map any input to the output.

One of the main reasons behind universal approximation is the activation function. Activation functions introduce nonlinear properties to the network. This helps the network learn any complex relationship between input and output.

### Recurrent Neural Network (RNN) – What is an RNN and why should you use it?
Let us first try to understand the difference between an RNN and an ANN from the architecture perspective :

A looping constraint on the hidden layer of ANN turns to RNN.      
          ![image](https://cdn.analyticsvidhya.com/wp-content/uploads/2020/02/assets_-LvBP1svpACTB1R1x_U4_-LwEQnQw8wHRB6_2zYtG_-LwEZT8zd07mLDuaQZwy_image-1.png)
          
As you can see here, RNN has a recurrent connection on the hidden state. This looping constraint ensures that sequential information is captured in the input data.

We can use recurrent neural networks to solve the problems related to:
- __Time Series data__
- __Text data__
- __Audio data__

### Convolution Neural Network (CNN) – What is a CNN and Why Should you use it?
Convolutional neural networks (CNN) are all the rage in the deep learning community right now. These CNN models are being used across different applications and domains, and they’re especially prevalent in image and video processing projects.

The building blocks of CNNs are filters a.k.a. kernels. Kernels are used to extract the relevant features from the input using the convolution operation. Let’s try to grasp the importance of filters using images as input data. Convolving an image with filters results in a feature map.

Though convolutional neural networks were introduced to solve problems related to image data, they perform impressively on sequential inputs as well.

__Advantages of Convolution Neural Network (CNN)__

CNN learns the filters automatically without mentioning it explicitly. These filters help in extracting the right and relevant features from the input data

CNN captures the spatial features from an image. Spatial features refer to the arrangement of pixels and the relationship between them in an image. They help us in identifying the object accurately, the location of an object, as well as its relation with other objects in an image.

CNN also follows the concept of parameter sharing. A single filter is applied across different parts of an input to produce a feature map.

## Comparing the Different Types of Neural Networks (MLP(ANN)  vs. RNN vs. CNN)
![image](https://cdn.analyticsvidhya.com/wp-content/uploads/2020/02/table.png)
