#!/usr/bin/env python3

# Simple neural network :

def NeuralNetwork(input, weight) :
    prediction = input * weight 
    return prediction

    
# ----------- Neural network with multiple inputs ---------------------

weights = [0.1, 0.2, 0]


def w_sum(a, b) :
    assert(len(a) == len(b)) 
    output = 0
    for i in range(len(a)) :
        output += (a[i] * b[i])
    return output

def neural_network(input, weights) :
    pred = w_sum(input, weights)
    return round(pred, 2)

if __name__ == '__main__' :
    # single input for simple neural network     
    weight = .1
    number_of_toes = [8.5, 9.5, 10, 9]
    input = number_of_toes[0]    # 8.5
    pred = NeuralNetwork(input, weight)
    print(round(pred, 2))
    # -----------------------------------
    # multiple inputs 
    toes = [8.5, 9.5, 9.9, 9.0]
    wlrec = [0.65, 0.8, 0.8, 0.9]
    nfans = [1.2, 1.3, 0.5, 1.0]
    input = [toes[0], wlrec[0], nfans[0]]
    pred = neural_network(input, weights)
    print(pred)

# $ python grokking-dl/ch3-simple-nn.py
# 0.85
# 0.98

# Inputs     Weights    Local 
#    |           |    Predictions
#    V           V    
# (8.50    *    0.1) =  0.85     =   toes prediction
# (0.65    *    0.2) =  0.13     =   wlrec prediction
# (1.20    *    0.0) =  0.00     =   fans prediction

# toes prediction + wlrec prediction + fans prediction = final prediction
#       0.85      +        0.13      +          0.00   = 0.98
