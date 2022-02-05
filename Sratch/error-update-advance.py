#!/usr/bin pypy3

# ------------------------ Simplest form of learning ------------------------

weight = .5
input = .5
target = .8
EPOCHS = 1101

step_amount = .001  # <= How much to move weights each iteration

for iteration in range(EPOCHS) :      # <= repeat the learning many times so that
                                    # error can keep getting smaller
    prediction = input * weight 
    error = (prediction - target) ** 2

    print(f"Error: {round(error, 3)}  prediction: {round(prediction, 3)}")

    # Try up
    up_prediction = input * (weight + step_amount)
    up_error = (target - up_prediction) ** 2

    # Try down 
    down_prediction = input * (weight - step_amount)
    down_error = (target - down_prediction) ** 2

    if (down_error < up_error) :
        weight = weight - step_amount
    if (down_error > up_error) :
        weight += step_amount


# $ pypy3 error-update-advance.py  
# Error: 0.301 Prediction:0.25
# Error: 0.301 Prediction:0.250
# ...........................
# Error: 0.0  prediction: 0.798      
# Error: 0.0  prediction: 0.799      
# Error: 0.0  prediction: 0.799      
# Error: 0.0  prediction: 0.8 
