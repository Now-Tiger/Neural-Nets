
# ---------------------------- Predictions on Mnist dataset : buidling neural nets : Class Methods ----------------------------

from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model, layers

import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print(len(tf.config.list_physical_devices('GPU')),'\n')


# Data parameters :
num_classes = 10
num_features = 784

# training paramters :
learning_rate = 0.1
training_steps = 2000
batch_size = 256
display_step = 100

# Network parameter :
n_hidden_1 = 128
n_hidden_2 = 256 

# prepare data :
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Covert to float 32 :
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape([-1, num_features])

# print(x_train.shape)
# (60000, 784)S

# print(x_train[0].shape)
# (784,)

# ------------------------------- Normalize images values from [0, 255] to [0, 1] -------------------------------
x_train, x_test = x_train/255., x_test / 255.

# --------------------------------- Use tf.data API to shuffle and batch data. ----------------------------------
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))

train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)


# ---------------------------------------- Create Tensorflow Model -----------------------------------------------

class NeuralNet(Model) :
    def __init__(self) :
        super(NeuralNet, self).__init__()
        # First fully connected hidden layer.
        self.fc1 = layers.Dense(n_hidden_1, activation = tf.nn.relu)
        # Second fully connected hidden layer.
        self.fc2 = layers.Dense(n_hidden_2, activation = tf.nn.relu)
        # Output layer :
        self.out = layers.Dense(num_classes, activation = tf.nn.softmax)

    def call(self, x, is_training = False) :
        x = self.fc1(x)
        x = self.out(x)
        if not is_training :
            x = tf.nn.softmax(x)
        return x

# Build neural network :
neural_net = NeuralNet()


# --------------------------------------------- cross-Entropy Loss ----------------------------------------------

# Note : This will apply softmax to logits.
def cross_entropy_loss(x, y) :
    # Convert labels into int64 for tf cross-entropy function.
    y = tf.cast(y, tf.int64)
    # Apply softmax to logits and compute cross-entropy
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = x)
    # Average loss across the batch :
    return tf.reduce_mean(loss)


# --------------------------------------------- Accuracy Metrix --------------------------------------------------

def accuracy(y_pred, y_true) :
    # Predicted class is index of height score in prediction vector (i.e argmax)
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis = -1)

# Stochastic optimizer descent optimizer :
optimizer = tf.optimizers.SGD(learning_rate)

# --------------------------------------------- Optimization process ---------------------------------------------

def run_optimization(x, y) :
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g :
        # --------------- Forward pass ------------------
        pred = neural_net(x, is_training = True)
        # Computing loss 
        loss = cross_entropy_loss(pred, y)
    # variable to update, i.e trainable_variables 
    trainable_variables = neural_net.trainable_variables
    # Computing gradients.
    gradients = g.gradient(loss, trainable_variables)
    # Update W and b following gradients 
    optimizer.apply_gradients(zip(gradients, trainable_variables))


# --------------------------------------------- Training ----------------------------------------------------- ---

for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1) :
    # Run optimization to update W and b :
    run_optimization(batch_x, batch_y)
    if step % display_step == 0 :
        pred = neural_net(batch_x, is_training = True)
        loss = cross_entropy_loss(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print(f'Step : {step}, loss : {loss}, accuracy : {acc}')


# ------------------------------------- Test model on validation set ---------------------------------------------

pred = neural_net(x_test, is_training = False) 
print(f'\nTest Accuracy : {accuracy(pred, y_test)}\n')

# ----------------------------------- Predict 5 images from validation dataset -----------------------------------

n_images = 5
test_images = x_test[:n_images]
predictions = neural_net(test_images)

# Display the images and model prediction :
for i in range(n_images):
    plt.imshow(np.reshape(test_images[i], [28, 28]), cmap='gray')
    plt.show()
    print(f'Model predicted : {np.argmax(predictions.numpy()[i])}')

# -----------------------------------------------------------------------------------------------------------------

# (tensor) D:\College\Data Science\Tensorflow\src>python tensorflow-mnist-classmethod.py
# 1 

# Step : 100, loss : 1.982330083847046, accuracy : 0.53515625
# Step : 200, loss : 1.831880807876587, accuracy : 0.6953125
# Step : 300, loss : 1.7254254817962646, accuracy : 0.83984375
# Step : 400, loss : 1.6411783695220947, accuracy : 0.88671875
# Step : 500, loss : 1.6201772689819336, accuracy : 0.89453125
# Step : 600, loss : 1.6346184015274048, accuracy : 0.875
# Step : 700, loss : 1.6048765182495117, accuracy : 0.8984375
# Step : 800, loss : 1.567501425743103, accuracy : 0.921875
# Step : 900, loss : 1.588587760925293, accuracy : 0.9140625
# Step : 1000, loss : 1.597954511642456, accuracy : 0.88671875
# Step : 1100, loss : 1.6013412475585938, accuracy : 0.87890625
# Step : 1200, loss : 1.562242865562439, accuracy : 0.93359375
# Step : 1300, loss : 1.6186699867248535, accuracy : 0.87109375
# Step : 1400, loss : 1.5562025308609009, accuracy : 0.9296875
# Step : 1500, loss : 1.5722897052764893, accuracy : 0.91796875
# Step : 1600, loss : 1.5456467866897583, accuracy : 0.92578125
# Step : 1700, loss : 1.5806275606155396, accuracy : 0.88671875
# Step : 1800, loss : 1.5617830753326416, accuracy : 0.921875
# Step : 1900, loss : 1.5347397327423096, accuracy : 0.9453125
# Step : 2000, loss : 1.578684687614441, accuracy : 0.89453125
#
# Test Accuracy : 0.9200000166893005
#
# Model predicted : 7
# Model predicted : 2
# Model predicted : 1
# Model predicted : 0
# Model predicted : 4
