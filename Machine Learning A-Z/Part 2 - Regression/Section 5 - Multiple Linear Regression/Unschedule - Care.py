# Multi Linear Regression


# 1) Importing Libraries

import numpy as np                 # Mathematics fucntopns
import matplotlib.pyplot as plt    # Plot and Charts
import pandas as pd                #Data Set management and import them
import seaborn as sns              # Covariance Matrix
#import FLASK


# 2) importing Dataset
startup_dataset = pd.read_csv('data.csv')
startup_X = startup_dataset.iloc[:,2:7].values   # dependend variable matrix means that will helo in prediction [rows , columns]
startup_Y = startup_dataset.iloc[:,7].values      # independent variable vector means that need to be predicted [rows , columns]



# 3)Taking Care of Missing Data (We take mean of that column and put in missing area thats simple)
from sklearn.preprocessing import Imputer  # importing class for data missing values
imputer = Imputer(missing_values = 'NaN' , strategy = 'mean' , axis = 0)   # making object of that class
imp_1 = imputer.fit(startup_X[:, 3:7])  # fit above imputer object to our matrix X (we can do this way as well to select column by index no by  imp_1 = imputer.fit(X[:, [1,2]])   )
startup_X[:,  3:7] = imp_1.transform(startup_X[:,  3:7])   # LHS X dataset and RHs imp_object to implement missing values  (   X[:, 1:3] = imp_1.transform(X[:, 1:3])   )





# 4) Categorical Data and Encoding it - in our case STATE is categorical variable
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
startup_X[:, 0]  = labelencoder_X.fit_transform(startup_X[:, 0])
#X = np.array(X,dtype= 'int64')
#Here we are splitting categorial variable states in column as no of catogories - no of columns
onehotencoder = OneHotEncoder(categorical_features = [0])
startup_X = onehotencoder.fit_transform(startup_X).toarray() # Dummy variable  is created against state


#labelencoder_Y = LabelEncoder()
#Y  = labelencoder_Y.fit_transform(Y)"""


# 5) Avoiding Dummy Variable Trap 
    # We always remove 1 column from dummy variable like in our case we generated three columns and dummy_var = dn-1
startup_X = startup_X[:, 1:]

# 6) Split Data into Train and Test
from sklearn.cross_validation import train_test_split
X_train , X_test , Y_train, Y_test = train_test_split(startup_X,startup_Y , test_size = 0.2 , random_state=0) #test_size can be 0.3 or 1/3
					

	
"""
 # 6) Feature Scaling
from sklearn.preprocessing import StandardScaler
std_X = StandardScaler()
X_train[:,1:3] = std_X.fit_transform(X_train[:,1:3])
X_test[:,1:3] = std_X.transform(X_test[:,1:3])
"""


# 7)  Fit MultiL Linear Regreesio to Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)



# 8) Predicting the Test Result Set
Y_pred = regressor.predict(X_test).astype(int)
X_test #sal_test is the predicted model salaries for employees



full = startup_X 
Y_PRED_FULL = regressor.predict(full).astype(int)

#================================================= Model Optimization ===============================================

# 9) Building a optimal model using BACKWARD ELIMINATION
## We will decide independent variables by its significance lavel and impact on responce/dependent variable




# Covariance Matrix (0 means both variables are completly independent)
data_corr= pd.read_csv('data.csv')
cm = data_corr.corr()
sns.heatmap(cm,square = True)
plt.yticks(rotation = 0 )
plt.xticks(rotation = 90)
plt.show()


# 9-1) Backward Elimination Preparation
#we are just making copy of of variableshere to test from zero
X1 = startup_X 
Y1 = startup_Y


# MLR Formula (y = b. + b1x1 +b2x2 .......bnXn) ,as in this case our library donot handle b. constant so we need to that care of that
# 1st way of append values but we need to manually change index in it
rows_count = X1.shape[0]  # no of rows
X1 = np.append(arr = X1 , values = np.ones((rows_count,1)).astype(float), axis = 1)
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
X1_optimized = X1[:,[0,1,2,3,4,5] ]# We will delete index step by step to get important predictors
regressor_OLS = sm.OLS(endog = Y1 , exog = X1_optimized).fit()  #Fit full model with all predictors
regressor_OLS.summary() # Check summary of model and remove highest P values predictors


X1_optimized = X1[:,[0,3,4,5,]]  # We will delete index step by step to get important predictors
regressor_OLS = sm.OLS(endog = Y1 , exog = X1_optimized).fit()  #Fit full model with all predictors
regressor_OLS.summary() 

X1_optimized = X1[:,[3,4,5]]  # We will delete index step by step to get important predictors
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

