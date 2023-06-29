"""
It is an simple liner regression where it is applied between the dependent variable,
and the independent variable where the predicition is done for the two variables!!
"""

#SINGLE LINEAR REGRESSION!!!
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

data = pd.read_excel('/content/Salary.xls')

X = data['YearsExperience'].values.reshape(-1, 1)
y = data['Salary'].values.reshape(-1,1)

linear = LinearRegression()

linear.fit(X, y)

y_pred = linear.predict(X)

coefficients = linear.coef_
intercept = linear.intercept_

print(y_pred)
print(coefficients)
print(intercept)

plt.scatter(x=X, y=y, label='Actual')
plt.plot(X, y_pred, color='red', label='Predicted')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.show()

"""
multiple linear regression it is appilied between the one dependent variable and more than one independent variable 
we can also use the mertrices to calculate effectiveness of the variables
"""
#MULTIPLE LINEAR REGRESSION!!

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_excel('/content/multipleregression.xlsx')

X = data['income'].values.reshape(-1, 1)
Y = data[['experience', 'age']].values.reshape(-1, 2)

linear = LinearRegression()
linear.fit(Y, X)

coefficients = linear.coef_
intercept = linear.intercept_
print(coefficients)
print(intercept)

y_pred = linear.predict(Y)

plt.scatter(X, y_pred,color='green',label='Predicted')
plt.plot(X,y_pred,color='red',label='Actual')
plt.xlabel("Actual Income")
plt.ylabel("Predicted Income")
plt.legend()
plt.show()
 
