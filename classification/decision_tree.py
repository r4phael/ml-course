# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 22:04:07 2019

@author: Jairo Souza
"""

# Importing the packages
from __future__ import absolute_import
from utils import plot_results_class, accuracy, f_measure, feature_scaling
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

# Importing the data
df = pd.read_csv('data/titanic.csv')

# Visualizing the dataset
df.describe()

# Visualizing the dataset
df = df.fillna(df.median())

# Defining the independent/dependent variables:
X = df.iloc[:, [5, 9]].values
y = df.iloc[:, 1].values

# Splitting the dataset in training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Scaling the features
X_train = feature_scaling(X_train)
X_test = feature_scaling(X_test)

# Fitting the Decisiont tree classifier with Training set
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the results with  Test set
y_pred = classifier.predict(X_test)

# Making the confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Metrics - accuracy using our functions and sklearn
accuracy(tp, fp, fn, tn)
classifier.score(X_test, y_test)

# Metrics - f1_score using our functions and sklearn
f_measure(tp, fp, fn)
f1_score(y_test, y_pred)  

# Plotting the training set results
plot_results_class(X_train, y_train, classifier, 'Decision Tree (Training set)')

# Plotting the test set results
plot_results_class(X_test, y_test, classifier, 'Decision Tree (Test set)')