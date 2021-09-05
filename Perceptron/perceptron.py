
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
    print('-'*50)
    print('Config : {}'.format(result.best_params_))
    print('-'*50)

    # Summarize all :
    means = result.cv_results_['mean_test_score']
    params = result.cv_results_['params']

    for mean, param in zip(means, params) :
        print('{} with : {}'.format(round(mean, 3),(param)))



# $ python perceptron.py
# Mean Accuracy : 0.857
# --------------------------------------------------
# Config : {'max_iter': 10}
# --------------------------------------------------
# 0.85 with : {'max_iter': 1}
# 0.857 with : {'max_iter': 10}
# 0.857 with : {'max_iter': 100}
# 0.857 with : {'max_iter': 1000}
# 0.857 with : {'max_iter': 10000}


