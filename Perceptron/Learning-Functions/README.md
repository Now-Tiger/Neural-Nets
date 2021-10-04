## Different 'Learning functions' in neural nets :
### Tanh

The hyperbolic tangent activation function is also referred to simply as the Tanh (also “tanh” and “TanH“) function.

It is very similar to the sigmoid activation function and even has the same S-shape.

The function takes any real value as input and outputs values in the range __-1 to 1.__ The __larger__ the input __(more positive)__, the __closer__ the output value will be to 1.0, whereas the __smaller__ the input __(more negative)__, the closer the output will be to -1.0.

The Tanh activation function is calculated as follows:

_(e^x – e^-x) / (e^x + e^-x)_

Where e is a mathematical constant that is the base of the natural logarithm.

![image](https://cdn.builtin.com/sites/www.builtin.com/files/styles/ckeditor_optimize/public/inline-images/9_activation%20functions%20deep%20learning.png)
```python
import numpy
from numpy import exp, dot, random, array

# A single layer neural net with 'tanh' activation fuction !!

class NeuralNetwork() :
    def __init__(self) :
        random.seed(2)
        self.synaptic_weights = random.random((3, 1)) - 1

    def __tanh(self, x) :
        return numpy.tanh(x)

    def __tanh_derivative(self, x) :
        t = (numpy.exp(x) - numpy.exp(-x)) / (numpy.exp(x) + numpy.exp(-x))
        return 1 - t**2

    def train(self, training_inputs, training_outputs, number_of_training_iterations) :
        # We train the neural network through a process of trial and error.
        # Adjusting the synaptic weights each time.
        for _ in range(number_of_training_iterations) :
            output = self.think(training_inputs)
            errors = training_outputs - output
            adjust = dot(training_inputs.T, errors * self.__tanh_derivative(output))
            self.synaptic_weights += adjust 

    def think(self, inputs) :
        return self.__tanh(dot(inputs, self.synaptic_weights))


if __name__ == '__main__' :
    neural_net = NeuralNetwork()
    print('Random starting synaptic weights :')
    print(neural_net.synaptic_weights)
    # The training set. We have 4 examples, each consisting of 3 input values
    # and 1 output value.
    training_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_output = array([[0, 1, 1, 0]]).T

    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    neural_net.train(training_inputs, training_output, 10000)

    print('\nNew synaptic weights after training :')
    print(neural_net.synaptic_weights)

    # Test the neural network with a new situation.
    print('\nConsidering new situation [1, 0, 0] -> ?: ')
    print(neural_net.think(array([1, 0, 0])))


# $ Alienware-Aw27 👘
# $ python neural-net-tanh.py 
# Random starting synaptic weights :
# [[-0.5640051 ] 
#  [-0.97407377] 
#  [-0.45033752]]

# New synaptic weights after training :
# [[ 5.39403241]
#  [-0.19472168]
#  [-0.34310257]]

# Considering new situation [1, 0, 0] -> ?: 
# [0.99995871]
```
- Sigmoid
- Relu
