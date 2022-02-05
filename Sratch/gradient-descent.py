#!/usr/bin pypy3

# What you're seeing here is the superior form of learning known as gradient descent.
# This method allows us to calculate both the direction and the amount you should change
# weight to reduce error.

weight = .5
input = .5
goal_pred = .8
EPOCHS = 30

for iteration in range(EPOCHS) :
    prediction = input * weight
    error = (prediction - goal_pred) ** 2
    direction_and_amount = (prediction - goal_pred) * input
    weight -= direction_and_amount
    print(f"Error: {round(error, 3)}  | Prediction: {round(prediction, 3)}")

# $ pypy gradient-descent.py
# Error: 0.303  | Prediction: 0.25
# Error: 0.17  | Prediction: 0.388
# Error: 0.096  | Prediction: 0.491
# Error: 0.054  | Prediction: 0.568
# Error: 0.03  | Prediction: 0.626
# Error: 0.017  | Prediction: 0.669
# Error: 0.01  | Prediction: 0.702
# Error: 0.005  | Prediction: 0.727
# Error: 0.003  | Prediction: 0.745
# Error: 0.002  | Prediction: 0.759
# Error: 0.001  | Prediction: 0.769
# Error: 0.001  | Prediction: 0.777
# Error: 0.0  | Prediction: 0.783
# Error: 0.0  | Prediction: 0.787
# Error: 0.0  | Prediction: 0.79
# Error: 0.0  | Prediction: 0.793
# Error: 0.0  | Prediction: 0.794
# Error: 0.0  | Prediction: 0.796
# Error: 0.0  | Prediction: 0.797
# Error: 0.0  | Prediction: 0.798
# Error: 0.0  | Prediction: 0.798
# Error: 0.0  | Prediction: 0.799
# Error: 0.0  | Prediction: 0.799
# Error: 0.0  | Prediction: 0.799
# Error: 0.0  | Prediction: 0.799
# Error: 0.0  | Prediction: 0.8
# Error: 0.0  | Prediction: 0.8
# Error: 0.0  | Prediction: 0.8
# Error: 0.0  | Prediction: 0.8
# Error: 0.0  | Prediction: 0.8
