# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 14:27:30 2019

@author: Jairo Souza
"""

"""
    #SEMANA 1
    Slide 8 -> limpeza dos dados: missing data (trocar valores NaN por media/mediana), normalização, quebrar as colunas, 
    plots de distribução dos dados; (pandas)
    
    Matriz de confusão-> implementar as funções; [vetor: tp, fp, tn, fn] -> 5 funçoes.
    fazer uma k-fold que mostre a divisão do conj. treinamento/teste (5-folds);
"""

# Importing the libraries
import pandas as pd

from __future__ import absolute_import

from utils import k_fold_cv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.model_selection


# Import the dataset
df = pd.read_csv('data/preprocessing_data.csv')

# Exploring the dataset
df.describe()

df.head(5)

# Distribution of nuimeric variables
df.boxplot()

df.hist()

# Fill NA with 0, median or mean.

df.fillna(df.mean())

df.fillna(df.median())

df.fillna(0)

df = df.fillna(df.median())

# Defining the dependend and dependent variables
X = df.iloc[:, :-1].values
y = df.iloc[:, 4].values

# Encoding the categorical values of dependent and independent variables

# encode labels with value between 0 and n_classes-1.
le = LabelEncoder()

# Fit and Transform the values
y = df.iloc[:, 4].values
y = le.fit_transform(y)

# Some machine learning techniques require you to drop one dimension from the representation so as 
# to avoid dependency among the variables. Use "drop_first=True" to achieve that.

# Independent variables:
X = pd.get_dummies(df.iloc[:, :-1] ,prefix=['city', 'sex'], drop_first=True).values
y = df.iloc[:, 4].values

# k-fold cross validation - 5 folds:
k_fold_cv(list(df.index.values))
 

# Using scikit-learn
kf = KFold(n_splits=5, random_state=42, shuffle=True)
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    #X_train, X_test = X[train_index], X[test_index]
    #y_train, y_test = y[train_index], y[test_index]


# Splitting the dataset 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Feature Scaling
sc = StandardScaler()
X = sc.fit_transform(X)