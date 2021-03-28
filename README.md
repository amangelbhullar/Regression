# Implementation in Python
1. Simple Linear Regression
2. Multiple Linear Regression
3. Polynomial Linear Regression
4. Decision Tree Regressor
5. Random Forest Regresor

# Simple_Linear_Regression

Simple linear regression is a statistical method that allows us to study relationships between two continuous (quantitative) variables.

1. variable denoted x, is regarded as the predictor, explanatory, or independent variable.
 
2. other variable, denoted y, is regarded as the response, outcome, or dependent variable.
 
### Import libraries and dataset

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

dataset = pd.read_csv("Salary_Data.csv")

x = dataset.iloc[:, :-1].values

y = dataset.iloc[:, -1].values

### Test and Train set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

### Linear Regression

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

### Visualization 

plt.scatter(x_train, y_train, color = 'red')

plt.plot(x_train, regressor.predict(x_train), color = 'blue')

plt.title("salary vs expeience (Training set)")

plt.xlabel("years of experience")

plt.ylabel("salary")

plt.show()

plt.scatter(x_test, y_test, color = "red")

plt.plot(x_train, regressor.predict(x_train), color = "blue")

plt.title("salary vs expeience (Training set)")

plt.xlabel("years of experience")

plt.ylabel("salary")

plt.show()


# Multi-Linear-Regression-SupervisedLearning

### Importing the libraries and dataset

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

dataset = pd.read_csv('50_Startups.csv')

X = dataset.iloc[:, :-1].values

y = dataset.iloc[:, -1].values

print(X)

### Encoding Categorical data

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')

X = np.array(ct.fit_transform(X))

print(x)

### Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

### Training the Multiple Linear Regression model on the Training set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)

### Predictions

y_pred = regressor.predict(X_test)

np.set_printoptions(precision=2)

print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

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

# desion_tree_regressor

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

dataset = pd.read_csv("yiyao_df3.csv")

x = dataset1.iloc[:, -5:-1].values

y = dataset1.iloc[:, -1].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state = 0)

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

np.set_printoptions(precision=2)

print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import r2_score

r2_score(y_test, y_pred)

#random_forest_regressor

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)

y_pred = regressor.predict(X_test)

np.set_printoptions(precision=2)

print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import r2_score

r2_score(y_test, y_pred)
regressor.fit(X_train, y_train)
