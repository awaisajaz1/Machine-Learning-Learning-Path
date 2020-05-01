# Simple Linear Regression

# 1) Importing Libraries

import numpy as np                 # Mathematics fucntopns
import matplotlib.pyplot as plt    # Plot and Charts
import pandas as pd                #Data Set management and import them

# !pip install pandas


# 2) importing Dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values   # dependend variable matrix means that will helo in prediction [rows , columns]
Y = dataset.iloc[:,1].values      # independent variable vector means that need to be predicted [rows , columns]
print(X)
print(Y)


# 3)Taking Care of Missing Data (We take mean of that column and put in missing area thats simple)
from sklearn.preprocessing import Imputer  # importing class for data missing values
imputer = Imputer(missing_values = 'NaN' , strategy = 'mean' , axis = 0)   # making object of that class
imp_1 = imputer.fit(X[:, 0:1])  # fit above imputer object to our matrix X (we can do this way as well to select column by index no by  imp_1 = imputer.fit(X[:, [1,2]])   )
X[:, 0:1] = imp_1.transform(X[:, 0:1])   # LHS X dataset and RHs imp_object to implement missing values  (   X[:, 1:3] = imp_1.transform(X[:, 1:3])   )



# 4) Categorical Data and Encoding it
"""from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0]  = labelencoder_X.fit_transform(X[:, 0])
X = np.array(X,dtype= 'int64')
#Here we are splitting categorial variable country in column as no of catogories - no of columns
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_Y = LabelEncoder()
Y  = labelencoder_Y.fit_transform(Y)"""



# 5) Split Data into Train and Test
from sklearn.cross_validation import train_test_split  ## it is deprecating
from sklearn.model_selection import train_test_split 
X_train , X_test , Y_train, Y_test = train_test_split(X,Y , test_size = 1/3 , random_state=0) #test_size can be 0.3 or 1/3
					

	
"""
 # 6) Feature Scaling
from sklearn.preprocessing import StandardScaler
std_X = StandardScaler()
X_train[:,1:3] = std_X.fit_transform(X_train[:,1:3])
X_test[:,1:3] = std_X.transform(X_test[:,1:3])
"""



# Fitting Simple Linear Regression to the Training Dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()  #we will use fit methot to fit the regressor object to our training sets below
regressor.fit(X_train , Y_train ) # Our SLR regressor learn the correlation of training set means no of experience and salary



# Determining the p value of predictor and responce variable
import statsmodels.api as sm
X2 = sm.add_constant(X)
est = sm.OLS(Y, X2)
est2 = est.fit()
print(est2.summary())


# Predicting the Test Result Set
sal_pred = regressor.predict(X_test) #sal_test is the predicted model salaries for employees



# Visualize the Training set results ... we already import visualization lib matplotlib.pyplot as plt
#we will plot orignal data on scatterplot chart
plt.scatter(X_train , Y_train, color = 'red')
plt.plot(X_train,regressor.predict(X_train) ,color = 'blue') #regressor.predict(X_train) tp predict the value of salary
plt.title('Salary vs Experience - (Train Dataset)')
plt.xlabel('Salary')
plt.ylabel('Emp Experience')
plt.show()




# Now on Test set will will fit our training set ml
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train,regressor.predict(X_train) ,color = 'blue') #regressor.predict(X_train)  predict the value of salary to y coordinate
plt.title('Salary vs Experience - (Test Dataset)')
plt.xlabel('Salary')
plt.ylabel('Emp Experience')
plt.show()




# Run model on full data set

full = X
salary = regressor.predict(full)


# Now on Test set will will fit our training set ml
plt.scatter(X, Y, color = 'red')
plt.plot(X,regressor.predict(full) ,color = 'blue') #regressor.predict(X_train)  predict the value of salary to y coordinate
plt.title('Salary vs Experience - (Full Dataset)')
plt.xlabel('Salary')
plt.ylabel('Emp Experience')
plt.show()




#Histogram
import seaborn as sns
sns.set()
plt.hist(Y,align='mid',normed=True,color = 'blue',bins=10)
plt.show()






