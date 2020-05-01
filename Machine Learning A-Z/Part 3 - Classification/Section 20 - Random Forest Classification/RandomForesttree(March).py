# Random ForestTree

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns              # Covariance Matrix


dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,1:4].values
Y = dataset.iloc[:,-1].values



## Missing Data 
#from sklearn.preprocessing import Imputer

#imputer = Imputer(missing_values='NaN', strategy = 'mean' ,  axis = 0)
#imp1 = imputer.fit(X[:,0:1])
#X[:,0:1] = imp1.transform(X[:,0:1])

# Covariance Matrix (0 means both variables are completly independent)
data_corr= pd.read_csv('Social_Network_Ads.csv')
cm = data_corr.corr()
sns.heatmap(cm,square = True)
plt.yticks(rotation = 0 )
plt.xticks(rotation = 90)
plt.show()

## Categorical Variablesinto Dummy variables
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])


X1 = X
Y1 =Y
# 9-2) Now we are going to start backward elimination
import statsmodels.formula.api as sm
X_optimized = X1[:,[0,1,2]]
X_optimized = X_optimized.astype(float)
regressor_ols = sm.OLS(endog = Y1,exog = X_optimized).fit()
regressor_ols.summary()

##Train and Test Splittion
from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y , test_size = 0.2 , random_state = 0)


## Feature Scalaing
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
Xtrain = sc_X.fit_transform(Xtrain)
Xtest = sc_X.transform(Xtest)
Xall = sc_X.transform(X)




## Fitting Random Forest 
from sklearn.ensemble import RandomForestClassifier
Classifier = RandomForestClassifier(n_estimators=1000 , criterion = 'entropy' , random_state = 0)
Classifier.fit(Xtrain,Ytrain)

##prediction
y_pred = Classifier.predict(Xtest)
#y_pred2 = Classifier.predict_proba(Xtest)


#confusion matrix
from sklearn.metrics import confusion_matrix
con_m = confusion_matrix(Ytest,y_pred)


### Predict against all
ypred_all = Classifier.predict(Xall)

#confusion matrix
from sklearn.metrics import confusion_matrix
con_mall = confusion_matrix(Y,ypred_all)
