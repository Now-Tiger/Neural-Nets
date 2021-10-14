## Develop a perceptron algorithm for class prediction.


``` python

# ------------------------------------------- The Perceptron Algorithm -------------------------------------------

import warnings
import seaborn as sns
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)
warnings.filterwarnings('ignore')

# Load the data :
data = datasets.load_breast_cancer()
X = data['data']
y = data['target']

# Before constructing the perceptron, let’s define a few helper functions. The sign function returns 1 for positive
# numbers and -1 for non-positive numbers.

# Next, the to_binary function can be used to convert predictions in {-1, +1} to their equivalents in {0, 1}, which is
# useful since the perceptron algorithm uses the former though binary data is typically stored as the latter. Finally,
# the standard_scaler standardizes our features, similar to scikit-learn’s StandardScaler.


def sign(a):
    return (-1)**(a < 0)

def to_binary(y):
    return y > 0

def standard_scaler(X):
    mean = X.mean(0)
    std = X.std(0)
    return (X-mean)/std

# The perceptron is implemented below.
# As usual, we optionally standardize and add an intercept term(Beta hat). Then we fit with the algorithm.
# This implementation tracks whether the perceptron has converged (i.e. all training algorithms are fitted correctly)
# and stops fitting if so. If not, it will run until n_iters is reached.

class Perceptron:
    def fit(self, X, y, n_iter = 10**3, lr = 0.001,
            add_intercept = True, standardize = True):
            
        # -------------- Add Info ---------------
        if standardize:
            X = standard_scaler(X)
        if add_intercept:
            ones = np.ones(len(X)).reshape(-1, 1)
        self.X = X
        self.y = y
        self.N, self.D = self.X.shape
        self.n_iter = n_iter
        self.lr = lr
        self.convergerd = False

        # ----------------- Fit -----------------
        beta = np.random.randn(self.D)/5
        for i in range(int(self.n_iter)):

            # -------------- Form Predictions --------------
            yhat = to_binary(sign(np.dot(self.X, beta)))

            # -------------- Check for convergence --------------
            if np.all(yhat == sign(self.y)):
                self.convergerd = True
                self.iterations_unitl_convergence = i
                break

            # -------------- Otherwise, adjust --------------
            for n in range(self.N):
                yhat_n = sign(np.dot(beta, self.X[n]))
                if (self.y[n] * yhat_n == -1):
                    beta += self.lr * self.y[n] * self.X[n]
                    
        # -------------- Retrun values --------------
        self.beta = beta
        self.yhat = to_binary(sign(np.dot(self.X, self.beta)))


if __name__ == '__main__' :
    percepton = Perceptron()
    percepton.fit(X, y, n_iter = 1e3, lr = 0.01)
    if percepton.convergerd :
        print(f'Converged after : {percepton.iterations_unitl_convergence} iterations')
    else :
        print('Not converged')

    print(np.mean(percepton.yhat == percepton.y))


# (tensor) D:\College\Data Science\Tensorflow\src>python perceptron-class-prediction.py
# Not converged
# 0.9771528998242531
```
