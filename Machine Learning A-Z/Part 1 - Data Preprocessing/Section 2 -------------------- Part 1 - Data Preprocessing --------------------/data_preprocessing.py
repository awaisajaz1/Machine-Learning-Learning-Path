# Data Preprocessing 
  
 # 1) Importing Libraries
 # 2) importing Dataset
 # 3) Missing Data
 # 4) Categorical Data
 # 5) Split Data into Train and Test
 # 6) Feature Scaling


# 1) Importing Libraries

import numpy as np                 # Mathematics fucntopns
import matplotlib.pyplot as plt    # Plot and Charts
import pandas as pd                #Data Set management and import them





# 2) importing Dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values   # dependend variable matrix means that will helo in prediction [rows , columns]
Y = dataset.iloc[:,3].values      # independent variable matrix means that need to be predicted [rows , columns]





# 3)Taking Care of Missing Data (We take mean of that column and put in missing area thats simple)
from sklearn.preprocessing import Imputer  # importing class for data missing values
imputer = Imputer(missing_values = 'NaN' , strategy = 'mean' , axis = 0)   # making object of that class
imp_1 = imputer.fit(X[:, 1:3])  # fit above imputer object to our matrix X (we can do this way as well to select column by index no by  imp_1 = imputer.fit(X[:, [1,2]])   )
X[:, 1:3] = imp_1.transform(X[:, 1:3])   # LHS X dataset and RHs imp_object to implement missing values  (   X[:, 1:3] = imp_1.transform(X[:, 1:3])   )





# 4) Categorical Data and Encoding it
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0]  = labelencoder_X.fit_transform(X[:, 0])
X = np.array(X,dtype= 'int64')
#Here we are splitting categorial variable country in column as no of catogories - no of columns
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_Y = LabelEncoder()
Y  = labelencoder_Y.fit_transform(Y)




# 5) Split Data into Train and Test
from sklearn.cross_validation import train_test_split
X_train , X_test , Y_train, Y_test = train_test_split(X,Y , test_size = 0.3 , random_state=0)
					

	

 # 6) Feature Scaling
from sklearn.preprocessing import StandardScaler
std_X = StandardScaler()
X_train = std_X.fit_transform(X_train[:,1:3])
X_test[:,1:3] = std_X.transform(X_test[:,1:3])
