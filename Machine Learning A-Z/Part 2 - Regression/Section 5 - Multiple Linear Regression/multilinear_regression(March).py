## Multi Linear Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd




dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,0:4].values
Y = dataset.iloc[:,4].values


## Missing Data 
#from sklearn.preprocessing import Imputer

#imputer = Imputer(missing_values='NaN', strategy = 'mean' ,  axis = 0)
#imp1 = imputer.fit(X[:,0:1])
#X[:,0:1] = imp1.transform(X[:,0:1])


## Categorical Variablesinto Dummy variables
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])

onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()


## Avoid Dummy Varible Trap
X = X[:,1:]


##Train and Test Splittion
from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y , test_size = 0.3 , random_state = 0)



## Fit Multiregrssion to Dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(Xtrain,Ytrain)


# Predict Result
ypred = regressor.predict(Xtest)



#================================================= Model Optimization ===============================================

# 9) Building a optimal model using BACKWARD ELIMINATION
## We will decide independent variables by its significance lavel and impact on responce/dependent variable



import seaborn as sns 
# Covariance Matrix (0 means both variables are completly independent)
data_corr= pd.read_csv('50_Startups.csv')
cm = data_corr.corr()
sns.heatmap(cm,square = True)
plt.yticks(rotation = 0 )
plt.xticks(rotation = 90)
plt.show()


# 9-1) Backward Elimination Preparation
#we are just making copy of of variableshere to test from zero
X1 = X 
Y1 = Y


# MLR Formula (y = b. + b1x1 +b2x2 .......bnXn) ,as in this case our library donot handle b. constant so we need to that care of that
# 1st way of append values but we need to manually change index in it , So we need to add constant value of 1 to last column and 
# that its index to 0
rows_count = X1.shape[0]  # no of rows
X1 = np.append(arr = X1 , values = np.ones((rows_count,1)).astype(int), axis = 1)
X1 = X1[:,[5,0,1,2,3,4]]
# 2ns way of append values in which (b.) value will be automaically set to 1 at index 0 (use any fron 1st or 2nd way)
X1 = np.append(arr = np.ones((rows_count,1)).astype(int) , values =X1 , axis = 1)



# 9-2) Now we are going to start backward elimination
import statsmodels.formula.api as sm
X_optimized = X1[:,[0,1,2,3,4,5]]
regressor_ols = sm.OLS(endog = Y1,exog = X_optimized).fit()
regressor_ols.summary()

X_optimized = X1[:,[0,1,3,4,5]]
regressor_ols = sm.OLS(endog = Y1,exog = X_optimized).fit()
regressor_ols.summary()


X_optimized = X1[:,[0,3,4,5]]
regressor_ols = sm.OLS(endog = Y1,exog = X_optimized).fit()
regressor_ols.summary()



X_optimized = X1[:,[0,3,5]]
regressor_ols = sm.OLS(endog = Y1,exog = X_optimized).fit()
regressor_ols.summary()


X_optimized = X1[:,[0,3]]
regressor_ols = sm.OLS(endog = Y1,exog = X_optimized).fit()
regressor_ols.summary()
