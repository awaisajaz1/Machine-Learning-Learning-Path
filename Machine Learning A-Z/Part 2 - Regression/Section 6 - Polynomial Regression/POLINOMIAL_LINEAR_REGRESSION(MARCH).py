### Polynomial Linear Regression


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,-1].values



### We are not splitting dataset as we dont have much data , we will test our ow data on trained model

## we are going to compare linear and polynimial model to see results

## Fitting Linear Regression Model
from sklearn.linear_model import LinearRegression

l_reg = LinearRegression()
l_reg.fit(X,Y)
l_reg.predict()

## Fitting Polynimial Linear Regression Model 
from sklearn.preprocessing import PolynomialFeatures

P_reg = PolynomialFeatures(degree = 2)
X_poly = P_reg.fit_transform(X)
l_reg2 = LinearRegression()
l_reg2.fit(X_poly,Y)



#Visulaze both models

#Linear
plt.scatter(X,Y,color = 'red')
plt.plot(X,l_reg.predict(X),color = 'blue')
plt.title("Truth or Bluff")
plt.xlabel("Position")
plt.ylabel("salary")
plt.show()


# Polynomial
plt.scatter(X,Y,color = 'red')
plt.plot(X,l_reg2.predict(P_reg.fit_transform(X)),color = 'blue')
plt.title("Truth or Bluff")
plt.xlabel("Position")
plt.ylabel("salary")
plt.show()



#prediction
#Linear
l_reg.predict(6.5)

#polynomial
l_reg2.predict(P_reg.fit_transform(6.5))




