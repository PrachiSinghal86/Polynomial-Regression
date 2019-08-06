# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 10:21:32 2019

@author: user
"""

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import dataset
dataset=pd.read_csv('Position_Salaries.Csv')
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#simple linear regressor
from sklearn.linear_model import LinearRegression
linreg=LinearRegression()
linreg.fit(X,y)
#Fitting polynomial regression to dataset
from sklearn.preprocessing import  PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)
linreg2=LinearRegression()
linreg2.fit(X_poly,y)
#visualizing linear regression
plt.scatter(X,y,color='red')
plt.plot(X,linreg.predict(X),color='blue')
plt.title('Salary Prediction')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
#visualizing polynomial regression
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid,linreg2.predict(poly_reg.fit_transform(X_grid)),color='blue')
plt.title('Salary Prediction')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

x=X_grid[55,0:1]
x=x.reshape(-1,1)

#Predicting linear regression
linreg.predict(x)
#Predicting polynomial regression
linreg2.predict(poly_reg.fit_transform(x))
