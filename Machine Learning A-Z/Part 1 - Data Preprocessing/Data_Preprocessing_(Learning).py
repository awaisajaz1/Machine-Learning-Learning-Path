############################################### Data Preprocessing

## Importing Libraries
import numpy as np     # Provides mathematics tools
import matplotlib.pyplot as plt  #Provides nice charts
import pandas as pd   # import and manage datasets


## import dataset using pandas 
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values



## Missing Data 
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values='NaN',strategy = 'mean' , axis = 0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])



## Categorical Variablesinto Dummy variables
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])

onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)


## Splitting Data into Traing and Test
from sklearn.cross_validation import train_test_split   # Itis deprecated so use below libraray
from sklearn.model_selection import train_test_split

Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y , test_size = 0.3 , random_state = 0)


## Feature Scalaing
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
Xtrain = sc_X.fit_transform(Xtrain)
Xtest = sc_X.transform(Xtest)




