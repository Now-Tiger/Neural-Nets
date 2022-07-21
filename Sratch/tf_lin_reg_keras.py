#!/usr/bin/env/ conda: 'tensor'
# -*- coding: utf-8 -*-

# Slightly better program than tf_linear_regression.py

import os
import time
import tensorflow as tf
from pprint import pprint
from warnings import filterwarnings

filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.random.set_seed(42)


class LinearRegression(tf.keras.Model):
    def __init__(self, num_params: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self._weights = tf.Variable(tf.random.uniform(
            (num_params, 1)), dtype=tf.float32)

    @tf.function
    def call(self, x):
        return tf.linalg.matmul(x, self._weights)


model = LinearRegression(5)


@tf.function
def train_step():
    with tf.GradientTape() as tape:
        y_hat = model(x)
        loss = tf.reduce_mean(tf.square(y - y_hat))
        gradients = tape.gradient(loss, model.variables)
    for g, v in zip(gradients, model.variables):
        v.assign_add(tf.constant([-.05], dtype=tf.float32) * g)
    return loss


if __name__ == "__main__":
    true_weights = tf.constant(list(range(5)), dtype=tf.float32)[:, tf.newaxis]
    x = tf.constant(tf.random.uniform((32, 5)), dtype=tf.float32)
    y = tf.constant(x @ true_weights, dtype=tf.float32)

    t0 = time.time()
    for i in range(1001):
        loss = train_step()
        if not (i % 200):
            print(f'mean squared loss at iteration {i:4d} is {loss:2.3f}')

    print('-'*50)
    pprint(model.variables)
    print('-'*50)
    print(f'time took: {time.time() - t0} seconds')

    print(model.summary())
    model.compile(loss='mse', metrics=['mae'])
    print(model.evaluate(x, y, verbose=1))


# (tensor) python tf_lin_reg_keras.py
# mean squared loss at iteration    0 is 26.432
# mean squared loss at iteration  200 is 0.035
# mean squared loss at iteration  400 is 0.003
# mean squared loss at iteration  600 is 0.000
# mean squared loss at iteration  800 is 0.000
# mean squared loss at iteration 1000 is 0.000
# --------------------------------------------------
# [<tf.Variable 'Variable:0' shape=(5, 1) dtype=float32, numpy=
# array([[-3.4998464e-03],
#        [ 1.0078143e+00],
#        [ 2.0056899e+00],
#        [ 3.0020447e+00],
#        [ 3.9896028e+00]], dtype=float32)>]
# --------------------------------------------------
# time took: 0.5868377685546875 seconds
# --------------------------------------------------
# Model: "linear_regression"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
# =================================================================
# Total params: 5
# Trainable params: 5
# Non-trainable params: 0
# _________________________________________________________________
# None
# --------------------------------------------------
# 1/1 [==============================] - 0s 128ms/step - loss: 9.6413e-06 - mae: 0.0025
# [9.641316864872351e-06, 0.0025263354182243347]
