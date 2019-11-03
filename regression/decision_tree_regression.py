# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 22:04:07 2019

@author: Jairo Souza
"""

# Importing the packages
from __future__ import absolute_import
from utils import plot_results_linear
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Importing the data
df = pd.read_csv('data/pricing_houses.csv')
df = df.loc[:, ['LotArea', 'PoolArea', 'GarageArea', 'OverallCond','YearBuilt', 'MSZoning', 'SalePrice']].sample(n=60, random_state=0, weights = 'SalePrice')
# df.to_csv('data/pricing_houses_small.csv')

# Visualizing the dataset
df.describe()

df.head(5)

# Defining the independent/dependent variables:
X = df.loc[:, ['LotArea']].values
y = df.loc[:, 'SalePrice'].values.reshape(-1,1)

# Splitting the dataset in training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Scaling the features
# X_train = feature_scaling(X_train)
# X_test = feature_scaling(X_test)

# Fitting the Simple Linear Regression with Training set
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)


# Metrics - score of regressor r^2
regressor.score(X, y)

# Predicting results from regressor
y_pred = regressor.predict(X)


# Comparing results from regressors: linear and poly
# Plotting the results of linear regression:
plot_results_linear(X, y, regressor, 'Linear Regression Results')