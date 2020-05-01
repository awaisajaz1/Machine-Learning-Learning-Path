# Simple Linear Regression


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values


## Missing Data 
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values='NaN', strategy = 'mean' ,  axis = 0)
imp1 = imputer.fit(X[:,0:1])
X[:,0:1] = imp1.transform(X[:,0:1])



from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y , test_size = 0.3 , random_state = 0)


##Fitting Linear Regression to Train Data

from sklearn.linear_model import LinearRegression

LRegressor = LinearRegression()

LRegressor.fit(Xtrain,Ytrain)


## Predict Test Data

ypred = LRegressor.predict(Xtest)





