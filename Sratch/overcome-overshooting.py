#!/usr/bin pypy3

# Gradient descent algorithm fundamentals with single input and target values.
# Understanding how derivatives work in subject of gradient descent.
# and update weights accordingly.

def gd_without_alpha(input, goal_pred, EPOCHS) -> None :
    """
        This algorithm is inefficient.
        Gradient overshoots if we had greater 
        input value than target. 
            e.g change input to int(5) 
                and we see chaos
    """
    weight = 0.0
    for iteration in range(EPOCHS) :
        pred = input * weight 
        error = (pred - goal_pred) ** 2
        delta = pred - goal_pred
        weight_delta = delta * input
        weight -= weight_delta
        print(f"{iteration + 1} | Error : {error:.2f} | Prediction : {round(pred, 2)}")

# ----------------------------------------------------------------------------------------

def gd_with_alpha(input, target, epoch, lr) -> None :
    """
        To overcome overshooting gradient problem 
        we introduce -
        alpha : learning rate.
    """
    weight = .0
    for iteration in range(epoch) :
        pred = input * weight
        error = (pred - target) ** 2
        derivative = (pred - target) * input 
        weight -=  (lr * derivative) 
        print(f"{iteration + 1} | Error : {error:.2f} | Prediction : {round(pred, 2)}")

if __name__ == '__main__' :
    goal_pred, input = (0.8, 2)
    EPOCH = 30
    gd_without_alpha(input, goal_pred, EPOCH)
    # This produces overshooting
    
    print("-"*50)
    
    weight = .0
    target = .8
    input = 2
    alpha = .1    # learning rate 
    gd_with_alpha(input, target, 20, alpha)
