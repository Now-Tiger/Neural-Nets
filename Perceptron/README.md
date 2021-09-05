# [Perceptron](https://en.wikipedia.org/wiki/Perceptron) 
![image](https://d2f0ora2gkri0g.cloudfront.net/dd/db/dddb807b-a15b-457d-a21a-8a9e6f029a3e.png)
The Perceptron is a linear machine learning algorithm for binary classification tasks.

It may be considered one of the first and one of the simplest types of artificial neural networks. It is definitely not “deep” learning but is an important building block.

Like logistic regression, it can quickly learn a linear separation in feature space for two-class classification tasks, although unlike logistic regression, it learns using the stochastic gradient descent optimization algorithm and does not predict calibrated probabilities.<br>

![image](https://miro.medium.com/max/390/1*3tZibzP1TPzpbGSjB8vKIg.png)

# [Perceptron Algorithm](https://ieeexplore.ieee.org/document/80230?arnumber=80230)

The Perceptron algorithm is a two-class (binary) classification machine learning algorithm.

It is a type of neural network model, perhaps the simplest type of neural network model.

It consists of a single node or neuron that takes a row of data as input and predicts a class label. This is achieved by calculating the weighted sum of the inputs and a bias (set to 1). The weighted sum of the input of the model is called the activation.

__Activation = Weights * Inputs + Bias__

If the activation is above 0.0, the model will output 1.0; otherwise, it will output 0.0


__Predict 1: If Activation > 0.0__

__Predict 0: If Activation <= 0.0__


Model weights are updated with a small proportion of the error each batch, and the proportion is controlled by a hyperparameter called the learning rate, typically set to a small value. This is to ensure learning does not occur too quickly, resulting in a possibly lower skill model, referred to as premature convergence of the optimization (search) procedure for the model weights.

__weights(t + 1) = weights(t) + learning_rate (expectedi – predicted) input_i__

Training is stopped when the error made by the model falls to a low level or no longer improves, or a maximum number of epochs is performed.

The learning rate and number of training epochs are hyperparameters of the algorithm that can be set using heuristics or hyperparameter tuning.


# A simple perceptron in python

``` python
# ---------------------------------------- Perceptron implementation ----------------------------------------

from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.linear_model import Perceptron


if __name__ == '__main__' :
    X, y = make_classification(n_samples = 1000, 
                               n_features = 10, n_informative = 10, 
                               n_redundant = 0, random_state = 1)

    model = Perceptron(eta0=0.0001)

    cv = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 3, random_state = 1)

    grid = dict()
    grid['max_iter'] = [1, 10, 100, 1000, 10000]

    search = GridSearchCV(model, grid, scoring = 'accuracy', cv = cv, n_jobs = -1)
    
    result = search.fit(X, y)

    # Summarize :
    print('Mean Accuracy : {}'.format(round(result.best_score_, 4)))
    print('-'*30)
    print('Config : {}'.format(result.best_params_))
    print('-'*30)

    # Summarize all :
    means = result.cv_results_['mean_test_score']
    params = result.cv_results_['params']

    for mean, param in zip(means, params) :
        print('{} with : {}'.format(round(mean, 3),(param)))


# $ python perceptron.py
# Mean Accuracy : 0.857
# ----------------------------------
# Config : {'max_iter': 10}
# ----------------------------------
# 0.85 with : {'max_iter': 1}
# 0.857 with : {'max_iter': 10}
# 0.857 with : {'max_iter': 100}
# 0.857 with : {'max_iter': 1000}
# 0.857 with : {'max_iter': 10000}


```
