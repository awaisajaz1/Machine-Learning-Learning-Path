# Logistic Regression 


import numpy as np                 # Mathematics fucntopns
import matplotlib.pyplot as plt    # Plot and Charts
import pandas as pd                #Data Set management and import them
import seaborn as sns              # Covariance Matrix


# 2) importing Dataset
DT_dataset = pd.read_csv('123.csv')
DT_X = DT_dataset.iloc[:,0:7].values   # dependend variable matrix means that will helo in prediction [rows , columns]
DT_Y = DT_dataset.iloc[:,7].values      # independent variable vector means that need to be predicted [rows , columns]


DT_Y2 = DT_dataset.iloc[:,2].values


# 3)Taking Care of Missing Data (We take mean of that column and put in missing area thats simple)
from sklearn.preprocessing import Imputer  # importing class for data missing values
imputer = Imputer(missing_values = 'NaN' , strategy = 'mean' , axis = 0)   # making object of that class
imp_1 = imputer.fit(DT_X[:, 1:3])  # fit above imputer object to our matrix X (we can do this way as well to select column by index no by  imp_1 = imputer.fit(X[:, [1,2]])   )
DT_X[:,  1:3] = imp_1.transform(DT_X[:,  1:3])   # LHS X dataset and RHs imp_object to implement missing values  (   X[:, 1:3] = imp_1.transform(X[:, 1:3])   )



# 4) Categorical Data and Encoding it - in our case STATE is categorical variable
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
DT_X[:, 6]  = labelencoder_X.fit_transform(DT_X[:, 6])
DT_X[:, 5]  = labelencoder_X.fit_transform(DT_X[:, 5])
DT_X[:, 4]  = labelencoder_X.fit_transform(DT_X[:, 4])
DT_X[:, 3]  = labelencoder_X.fit_transform(DT_X[:, 3])
DT_X[:, 2]  = labelencoder_X.fit_transform(DT_X[:, 2])
DT_X[:, 1]  = labelencoder_X.fit_transform(DT_X[:, 1])

DT_Y  = labelencoder_X.fit_transform(DT_Y)
#X = np.array(X,dtype= 'int64')
#Here we are splitting categorial variable states in column as no of catogories - no of columns
onehotencoder = OneHotEncoder(categorical_features = [3])
DT_X = onehotencoder.fit_transform(DT_X).toarray() # Dummy variable  is created against state



for i in DT_X:
    print(i)
    

# 5) Now we are going to start backward elimination
#we are just making copy of of variableshere to test from zero
X1 = DT_X
Y1 = DT_Y

import statsmodels.formula.api as sm
# we are going to make a varibale of collection of independent/predictors
X1 = np.matrix(DT_X , dtype = 'float')
X1_optimized = X1  # We will delete index step by step to get important predictors
regressor_OLS = sm.OLS(endog = Y1, exog = X1_optimized).fit()  #Fit full model with all predictors
regressor_OLS.summary() 


X1_optimized = X1[ : , [0,1,2,3,5,6]]  # We will delete index step by step to get important predictors
regressor_OLS = sm.OLS(endog = Y1, exog = X1_optimized).fit()  #Fit full model with all predictors
regressor_OLS.summary() 



# 6) Split Data into Train and Test
from sklearn.cross_validation import train_test_split
X_train , X_test , Y_train, Y_test = train_test_split(X1,Y1 , test_size = 0.3 , random_state=0) #test_size can be 0.3 or 1/3



 # 7) Feature Scaling
from sklearn.preprocessing import StandardScaler
std_X = StandardScaler()
X_train[:,0:2] = std_X.fit_transform(X_train[:,0:2])
X_test[:,0:2] = std_X.transform(X_test[:,0:2])



# 8) Fitting Logistic Regression to Training Set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train,Y_train)


# 8) Fitting KNN to Training Set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5 , metric = 'minkowski' , p = 2 )
classifier.fit(X_train,Y_train)


# 9) Predicting the Test Result
y_pred = classifier.predict(X_test)


#confusion matrix
from sklearn.metrics import confusion_matrix
con_m = confusion_matrix(Y_test,y_pred)























x = 5
total = 0
while x > 0:
    if total == 10:
        break
    total = total + x
    print(total)
    
    
    
    x = x // 2
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
