#!/usr/bin/env/ conda:'tensor'
# -*- coding: utf-8 -*-

# custom class for using tensorflow Linear Regression with tf.keras.Model
import os
import time
import tensorflow as tf
from pprint import pprint
from warnings import filterwarnings

filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.random.set_seed(42)


class LinearRegression(object):
    def __init__(self, num_params) -> None:
        self._weights = tf.Variable(tf.random.uniform(
            (num_params, 1)), dtype=tf.float32)

    @tf.function
    def __call__(self, x):
        return tf.linalg.matmul(x, self._weights)

    @property
    def variables(self):
        return self._weights


if __name__ == "__main__":
    true_weights = tf.constant(list(range(5)), dtype=tf.float32)[:, tf.newaxis]
    x = tf.constant(tf.random.uniform((32, 5)), dtype=tf.float32)
    y = tf.constant(x @  true_weights, dtype=tf.float32)

    model = LinearRegression(5)

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            y_hat = model(x)
            loss = tf.reduce_mean(tf.square(y - y_hat))
        gradients = tape.gradient(loss, model.variables)
        model.variables.assign_add(tf.constant(
            [-.05], dtype=tf.float32) * gradients)
        return loss

    t0 = time.time()
    for i in range(1001):
        loss = train_step()
        if not (i % 200):
            print(f'mean squared loss at iteration {i:4d} is {loss:2.3f}')
    print('-'*50)
    pprint(model.variables)
    print(f'time took: {time.time() - t0} seconds')


# (tensor) python tf_linear_regression.py
# mean squared loss at iteration    0 is 15.1177
# mean squared loss at iteration  200 is 0.0278
# mean squared loss at iteration  400 is 0.0012
# mean squared loss at iteration  600 is 0.0001
# mean squared loss at iteration  800 is 0.0000
# mean squared loss at iteration 1000 is 0.0000
# ------------------------------------------------------------
# <tf.Variable 'Variable:0' shape=(5, 1) dtype=float32, numpy=
# array([[-1.9201766e-03],
#        [ 1.0043812e+00],
#        [ 2.0000753e+00],
#        [ 3.0021193e+00],
#        [ 3.9955370e+00]], dtype=float32)>
# time took: 0.5535032749176025 seconds
