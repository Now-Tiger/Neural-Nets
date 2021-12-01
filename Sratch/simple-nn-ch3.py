#!/usr/bin/env pypy3

from timeit import timeit
start = timeit()

weight = .1 

def neuralNet(input, weight) -> None:
    predictions = (input * weight)
    return predictions

num_toes = [8.5, 9.5, 10, 9.0]
input = num_toes[0]
print(round(neuralNet(input, weight), 3),'%')

# ---------------------------------------------------------
print('-'*6)

# Making predictions using multiple inputs

weights = [.1, .2, 0] 

def net(input, weights) :
    preds = weighted_sum(input, weights)
    return preds

def weighted_sum(a, b) :
    assert( len(a) == len(b) ) 
    output = 0
    for i in range(len(a)) :
        output += (a[i] * b[i])
    return output

toes = [8.5, 9.5, 9.9, 9.0]
wlrec = [.65, .8, .8, .9]
nfans = [1.2, 1.3, 0.5, 1.0]

""" ------------------------------------------------------
        Above dataset is the current status at the beginning
    of each game for the first four games in a season:
    toes = current number of toes
    wlrec = current games won (percent)
    nfans = fan count (in millions)
    ------------------------------------------------------
"""

input = [toes[0], wlrec[0], nfans[0]]
print(round(net(input, weights), 3),'%')

print('-'*6)


# -----------------------------------------------------------
#
# Numpy code 

import numpy as np

weights = np.array([0.1, .2, 0])

def neural_net(input, weights) :
    preds = input.dot(weights)
    return preds 

toes = np.array([8.5, 9.5, 9.9, 9.0])
wlrec = np.array([.65, .8, .8, .9])
nfans = np.array([1.2, 1.3, 0.5, 1.0])

input = np.array([toes[0], wlrec[0], nfans[0]])
print(round(neural_net(input, weights), 3),'%')


end = timeit()
print(f"Time taken : {round(end - start, 3)}")