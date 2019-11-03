# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 22:04:07 2019

@author: Jairo Souza
"""

# Importing the packages
from __future__ import absolute_import
from utils import plot_results_linear, plot_results_poly
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

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
regressor = LinearRegression()
regressor.fit(X, y)


# Fitting the Polynomial Regression with Training set
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)

lin_reg_poly = LinearRegression()
lin_reg_poly.fit(X_poly, y)

# Metrics - score of regressor r^2
lin_reg_poly.score(X_poly, y)

# Predicting results from regressor
y_pred = lin_reg_poly.predict(X_poly)


# Comparing results from regressors: linear and poly
# Plotting the results of linear regression:
plot_results_linear(X, y, regressor, 'Linear Regression Results')