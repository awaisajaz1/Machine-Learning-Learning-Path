# Polinomial Linear Regression

# 1) Importing Libraries

import numpy as np                 # Mathematics fucntopns
import matplotlib.pyplot as plt    # Plot and Charts
import pandas as pd                #Data Set management and import them
import seaborn as sns              # Covariance Matrix



# 2) importing Dataset
poli_dataset = pd.read_csv('Position_Salaries.csv')
X = poli_dataset.iloc[:,1:2].values   # dependend variable matrix means that will helo in prediction [rows , columns]
Y = poli_dataset.iloc[:,2].values      # independent variable vector means that need to be predicted [rows , columns]
#poli_dataset.iloc[:,1:2].values  we specify 1:2 in it to make it matrix ,upper bound will not be included so relax



# 3)Taking Care of Missing Data (We take mean of that column and put in missing area thats simple)
from sklearn.preprocessing import Imputer  # importing class for data missing values
imputer = Imputer(missing_values = 'NaN' , strategy = 'mean' , axis = 0)   # making object of that class
imp_1 = imputer.fit(X[:, 0:1])  # fit above imputer object to our matrix X (we can do this way as well to select column by index no by  imp_1 = imputer.fit(X[:, [1,2]])   )
X[:,  0:1] = imp_1.transform(X[:, 0:1])   # LHS X dataset and RHs imp_object to implement missing values  (   X[:, 1:3] = imp_1.transform(X[:, 1:3])   )


"""
# 4) Categorical Data and Encoding it - in our case STATE is categorical variable
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
startup_X[:, 3]  = labelencoder_X.fit_transform(startup_X[:, 3])
#X = np.array(X,dtype= 'int64')
#Here we are splitting categorial variable states in column as no of catogories - no of columns
onehotencoder = OneHotEncoder(categorical_features = [3])
startup_X = onehotencoder.fit_transform(startup_X).toarray() # Dummy variable  is created against state

#labelencoder_Y = LabelEncoder()
#Y  = labelencoder_Y.fit_transform(Y)



# 5) Avoiding Dummy Variable Trap 
    # We always remove 1 column from dummy variable like in our case we generated three columns and dummy_var = dn-1
startup_X = startup_X[:, 1:]
"""



#NOTE We have very small data , to predict very accurate result we are not splitting our dataset instead we are going with full -Skipping this step
# 6) Split Data into Train and Test
from sklearn.cross_validation import train_test_split
X_train , X_test , Y_train, Y_test = train_test_split(X,Y , test_size = 0.2 , random_state=0) #test_size can be 0.3 or 1/3


# Plotting data into chart to see curve
plt.scatter(X , Y, color = 'red') #regressor.predict(X_train) tp predict the value of salary
plt.title('Polinomial Regression')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()


# 6) Fitting Linear Regression to Dataset    (It is just made to compare)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg .fit(X,Y)




# 7) Fitting Polynomial Regression to Dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression() 	
lin_reg2 .fit(X_poly,Y)	




# 8 ) Visualization

# Linear Regression
plt.scatter(X, Y, color = 'red')
plt.plot(X,lin_reg.predict(X) ,color = 'blue') #regressor.predict(X) tp predict the value of salary
plt.title('Linear Regression')
plt.xlabel('Exp position')
plt.ylabel('Salary')
plt.show() # If we see result it is not goog model to show




# Polynomial Linear Regression
plt.scatter(X, Y, color = 'red')
plt.plot(X,lin_reg2.predict(X_poly) ,color = 'blue') #Ploynomial regressor.predict(X) tp predict the value of salary
plt.title('Polynimial Regression')
plt.xlabel('Exp position')
plt.ylabel('Salary')
plt.show()



# Polynomial Linear Regression (better visulaization)
x_grid = np.arange(min(X),max(X),0.1)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(X, Y, color = 'red')
plt.plot(x_grid,lin_reg2.predict(poly_reg.fit_transform(x_grid)) ,color = 'blue') #Ploynomial regressor.predict(X) tp predict the value of salary
plt.title('Polynimial Regression')
plt.xlabel('Exp position')
plt.ylabel('Salary')
plt.show()


# 9) Predictions Result
# Linear Regression Prediction
pos_pred = lin_reg.predict(X) 
pos_pred1 = lin_reg.predict(6.5) 

# Polunomial Regression Prediction
pos_pred2  = lin_reg2.predict(X_poly)
pos_pred3  = lin_reg2.predict(poly_reg.fit_transform(6.5))