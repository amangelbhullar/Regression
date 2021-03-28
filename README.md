# Polynomial_Linear_Regression
## Importing the libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

##Importing the dataset

dataset = pd.read_csv("Position_Salaries.csv")

x = dataset.iloc[:, 1:-1].values

y = dataset.iloc[:, -1].values

## Training the Linear Regression model on the whole dataset

from sklearn.linear_model import LinearRegression

linear_regressor = LinearRegression()

linear_regressor.fit(x, y)

LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)

## Training the Polynomial Regression model on the whole dataset

from sklearn.preprocessing import PolynomialFeatures

poly_r = PolynomialFeatures(degree = 4)

x_poly = poly_r.fit_transform(x)

lin_reg2 = LinearRegression()

lin_reg2.fit(x_poly, y)

LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)

## Visualising the Linear Regression results

plt.scatter(x, y, color = "red")

plt.plot(x, linear_regressor.predict(x), color = "blue")

plt.title('position level vs salary')

plt.xlabel('position level')

plt.ylabel('salary')

plt.show()

## Visualising the Polynomial Regression results

plt.scatter(x, y, color = "red")

plt.plot(x, lin_reg2.predict(x_poly), color = "blue")

plt.title('position level vs salary')

plt.xlabel('position level')

plt.ylabel('salary')

plt.show()

## Predicting a new result with Linear Regression

linear_regressor.predict([[6.5]])


## Predicting a new result with Polynomial Regression


lin_reg2.predict(poly_r.fit_transform([[6.5]]))
