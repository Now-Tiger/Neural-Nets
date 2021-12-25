#!/usr/bin pypy3

# ---------------------------------- Compare + Learn = Update ----------------------------------
# Compare: Does your network make
# good predictions?
# Let’s measure the error and find out!

knob_weight = .5
input = .5
goal_pred = .8 

pred = input * knob_weight
error = (pred - goal_pred) ** 2
print(f"error : {round(error, 3)}")   # -> 0.303
print('-'*15)


# ------------------- Simplest form of neural network -------------------

weight = .1 
lr = .1     # learning rate

def neuralNet(input, weight) :
    prediction = input * weight
    return round(prediction, 3)

number_of_toes = [8.5]
win_or_lose_binary = [1]      # 1:Won else 0:Lose

input = number_of_toes[0]
true = win_or_lose_binary[0]

pred = neuralNet(input, weight)
error = (pred - true) ** 2 
print(f"error : {round(error, 3)}")     # 0.023   

print('-'*15)


# --------------- Making a prediction with a higher weight and evaluating error ---------------

lr = .1 
p_up = neuralNet(input, weight + lr)
error_up = (p_up - true) ** 2
print(f"updated error : {round(error_up, 3)}")

print('-'*15)


# -------------------------- Make predictions on lower learning rate --------------------------

lr = .01
p_dn = neuralNet(input, weight + lr)
error_dn = (p_dn - true) ** 2
print(f"updated error : {round(error_dn, 3)}")  # 0.004  <- best updated

# $ pypy3 ch3-gradient-descent.py 
# error : 0.303
# ---------------      
# error : 0.023        
# ---------------      
# updated error : 0.49 
# ---------------      
# updated error : 0.004
