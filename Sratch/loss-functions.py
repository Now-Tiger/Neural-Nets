#!/usr/bin/ pypy3

from __future__ import annotations
from typing import List
import math

# 1. Mean Squared Error (MSE) 

def mean_squared_error(actuals: List[int | float], 
                       predicted: List[int | float]) -> None :
    sum_square_error: float = .0
    for i in range(len(actuals)) :
        sum_square_error += (actuals[i] - predicted[i]) ** 2
    mean_square_error: float = 1.0/len(actuals) * sum_square_error
    return mean_square_error

actuals = [2, 4, 6, 8, 10]
predicted = [.1, 2, 1, 3, 2.4]
print(mean_squared_error(actuals, predicted)) 

# --------------------------------------------------------------------

# 2. Cross Entropy 
#    2.1. Binary Cross Entropy

def binary_cross_entropy(actuals: List[int], 
                         predicted: List[int | float]) -> None : 
    sum_score: float = .0
    for i in range(len(actuals)) :
        sum_score += actuals[i] * math.log(1e-15 + predicted[i])
    mean_sum_score: float = 1.0/len(actuals) * sum_score
    return -mean_sum_score

actuals = [1, 0, 0, 1, 0, 0]
predicted = [.1, 1, 1, 0, 1, 0]
print(binary_cross_entropy(actuals, predicted))


#   2.2 Categorical Cross Entropy

def categorical_cross_entropy(actuals: List[int], 
                              predicted: List[int | float]) -> None :
    sum_score: float = .0
    for i in range(len(actuals)) :
        for j in range(len(actuals)) :
            sum_score += actuals[i][j] * math.log(1e-15 + predicted[i][j])
    mean_sum_score: float = 1.0/len(actuals) * sum_score
    return -mean_sum_score
