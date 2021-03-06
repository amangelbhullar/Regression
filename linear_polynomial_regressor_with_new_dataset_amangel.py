# -*- coding: utf-8 -*-
"""linear polynomial regressor with new dataset amangel.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1l735SWKSFil7TADLLvcVOmGG83YCL9bj

## import libraries and dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("yiyao_df3.csv")

print(dataset)

dataset.head()

"""# rearrange column"""

df_reorder = dataset[['Gender','Ethnicity', 'DevType',  'Hobbyist', 'Employment', 'Country', 'EdLevel', 'UndergradMajor', 'OrgSize', 'Year', 'Age',  'LanguageWorkedWith', 'DatabaseWorkedWith', 'YearsCodePro', 'ConvertedComp']] # rearrange column here
df_reorder.to_csv('reorder.csv', index=False)

dataset1 = pd.read_csv("reorder.csv")

dataset1.head()

"""# split dataset"""

x = dataset1.iloc[:, -2:-1].values
y = dataset1.iloc[:, -1].values

print(x)

print(y)

x = np.nan_to_num(x)
y = np.nan_to_num(y)

"""## Splitting the dataset into the Training set and Test set"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

"""# model building- polynomial linear regression"""

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X_train)
regressor = LinearRegression()
regressor.fit(X_poly, y_train)

"""## Predicting a new result with polynomial Linear Regression"""

y_pred = regressor.predict(poly_reg.transform(X_test))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

"""## Evaluation"""

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

