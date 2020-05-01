# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns              # Covariance Matrix

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]    # We are deleting one dummy variabke column


# Determining the p value of predictor and responce variable
import statsmodels.api as sm
X2 = sm.add_constant(X)
est = sm.OLS(Y, X2)
est2 = est.fit()
print(est2.summary())

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)















#================================================= Model Optimization ===============================================

# 9) Building a optimal model using BACKWARD ELIMINATION
## We will decide independent variables by its significance lavel and impact on responce/dependent variable




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
Y1 = y


# MLR Formula (y = b. + b1x1 +b2x2 .......bnXn) ,as in this case our library donot handle b. constant so we need to that care of that
# 1st way of append values but we need to manually change index in it , So we need to add constant value of 1 to last column and 
# that its index to 0
rows_count = X1.shape[0]  # no of rows
X1 = np.append(arr = X1 , values = np.ones((rows_count,1)).astype(int), axis = 1)
X1 = X1[:,[5,0,1,2,3,4]]
# 2ns way of append values in which (b.) value will be automaically set to 1 at index 0 (use any fron 1st or 2nd way)
X1 = np.append(arr = np.ones((rows_count,1)).astype(int) , values =X1 , axis = 1)

"""
for i in X1:
     X2 = X1[:,[5]] /2
"""

# 9-2) Now we are going to start backward elimination
import statsmodels.formula.api as sm
# we are going to make a varibale of collection of independent/predictors
X1_optimized = X1[:,[0,1,2,3,4,5]]  # We will delete index step by step to get important predictors
regressor_OLS = sm.OLS(endog = Y1 , exog = X1_optimized).fit()  #Fit full model with all predictors
regressor_OLS.summary() # Check summary of model and remove highest P values predictors


X1_optimized = X1[:,[0,2,3,4,5,]]  # We will delete index step by step to get important predictors
regressor_OLS = sm.OLS(endog = Y1 , exog = X1_optimized).fit()  #Fit full model with all predictors
regressor_OLS.summary() 

X1_optimized = X1[:,[0,3,5]]  # We will delete index step by step to get important predictors
regressor_OLS = sm.OLS(endog = Y1 , exog = X1_optimized).fit()  #Fit full model with all predictors
regressor_OLS.summary() 

X1_optimized = X1[:,[0,3]]  # We will delete index step by step to get important predictors
regressor_OLS = sm.OLS(endog = Y1 , exog = X1_optimized).fit()  #Fit full model with all predictors
regressor_OLS.summary() 




"""
SL = 0.01
import statsmodels.formula.api as sm
def backwardElimination(X1, SL):
    numVars = len(X1[0]) # Counting no of columns
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Y1, X1).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(X1, j, 1)
    regressor_OLS.summary()
    return x
 

X_opt = X1[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)


"""

