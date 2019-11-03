# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 22:04:07 2019

@author: Jairo Souza
"""

# Importing the packages
from __future__ import absolute_import
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.regression.linear_model as sm

# Importing the data
df = pd.read_csv('data/pricing_houses.csv')
df = df.loc[:, ['LotArea', 'PoolArea', 'GarageArea', 'OverallCond','YearBuilt', 'MSZoning', 'SalePrice']].sample(n=60, random_state=0, weights = 'SalePrice')
# df.to_csv('data/pricing_houses_small.csv')

# Visualizing the dataset
df.describe()

df.head(5)

# Encoding the categorical features and avoiding the Dummy variable trap
df = pd.get_dummies(df , columns = ['MSZoning'], drop_first=True)

# Defining the independent/dependent variables:
X = df[df.columns[~df.columns.isin(['SalePrice'])]].values
y = df['SalePrice'].values.reshape(-1,1)

# Splitting the dataset in training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Scaling the features
# X_train = feature_scaling(X_train)
# X_test = feature_scaling(X_test)

# Fitting the Multiple Linear Regression with Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting results from regressor
y_pred = regressor.predict(X_test)

# Metrics - score of regressor r^2
regressor.score(X_test, y_test)


# Backaward Elimination:
X = np.append(arr = np.ones((60,1)).astype(int), values = X, axis =1)
X_opt = X[:, [0,1,2,3,4,5,6,8,9]]
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
regressor_ols.summary()