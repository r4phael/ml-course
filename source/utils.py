# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 10:25:30 2019

@author: Jairo Souza
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler


# Function to return k_fold_cv values
def k_fold_cv(indexes, k = 5, seed = 42):
    
    size = len(indexes)
    subset_size = round(size / k)
    random.Random(seed).shuffle(indexes)
    subsets = [indexes[x:x+subset_size] for x in range(0, len(indexes), subset_size)]
    kfolds = []
    for i in range(k):
        test = subsets[i]
        train = []
        for subset in subsets:
            if subset != test:
                train.append(subset)
        kfolds.append((train,test))
        
    return kfolds

# accuracy of models
def accuracy (tp, fp, fn, tn):
    accuracy = ((tp + tn) / (tp + tn + fp + fn))
    return (accuracy)
    

# precision or positive predictive value (PPV)
def precision (tp, fp):
    precision =  (tp / (tp + fp))
    return precision

# sensitivity, recall, hit rate, or true positive rate (TPR)
def recall(tp, fn):
    recall =  (tp / (tp + fn))
    return recall

# is the harmonic mean of precision and sensitivity
def f_measure(tp, fp, fn):
    f_measure = (2 * precision(tp, fp) * recall(tp, fn)) / (recall(tp, fn) + precision(tp, fp))
    return f_measure

# specificity, selectivity or true negative rate (TNR)
def true_neg_rate(fp, tn):
    true_neg_rate = (tn / (tn + fp))
    return true_neg_rate

#  negative predictive value (NPV)
def neg_pred_value(fn, tn):
    neg_pred_value = (tn / (tn + fn))
    return neg_pred_value

# Informedness or Bookmaker Informedness (BM)
def informedness(tp, fp, fn, tn):
    inform = (recall(tp, fn) + true_neg_rate(tn, fp)) - 1
    return inform

# Markedness (MK)
def markdness(tp, fp, fn, tn):    
    mark = (precision(tp, fp) + neg_pred_value(tn, fn)) - 1
    return mark

# feature scaling 
def feature_scaling(data):
    sc = StandardScaler()
    return sc.fit_transform(data)

# Plotting the Classification results 
def plot_results_class(X, y, classifier, title):
    X_set, y_set = X, y
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title(title)
    plt.xlabel('Age')
    plt.ylabel('Fare')
    plt.legend()
    plt.show()
    
# Plotting the Regression results
def plot_results_linear(X, y, regressor, title):
    plt.scatter(X, y, color = 'red')
    plt.plot(X, regressor.predict(X), color = 'blue')
    plt.title(title)
    plt.xlabel('Lot Area')
    plt.ylabel('Sale Price')
    plt.show()
 
# Visualising the Polynomial Regression results    
def plot_results_poly(X, y, lin_reg_poly, poly_reg, title):
    plt.scatter(X, y, color = 'red')
    plt.plot(X, lin_reg_poly.predict(poly_reg.fit_transform(X)), color = 'blue')
    plt.title(title)
    plt.xlabel('Lot Area')
    plt.ylabel('Sale Price')
    plt.show()    
        