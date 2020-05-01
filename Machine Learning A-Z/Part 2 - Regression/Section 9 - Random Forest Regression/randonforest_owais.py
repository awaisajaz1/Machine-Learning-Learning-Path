# Random Forest Regression

import numpy as np                 # Mathematics fucntopns
import matplotlib.pyplot as plt    # Plot and Charts
import pandas as pd                #Data Set management and import them
import seaborn as sns              # Covariance Matrix


# 2) importing Dataset
DT_dataset = pd.read_csv('Position_Salaries.csv')
DT_X = DT_dataset.iloc[:,1:2].values   # dependend variable matrix means that will helo in prediction [rows , columns]
DT_Y = DT_dataset.iloc[:,2].values      # independent variable vector means that need to be predicted [rows , columns]
DT_Y2 = DT_dataset.iloc[:,2].values


# 3)Taking Care of Missing Data (We take mean of that column and put in missing area thats simple)
from sklearn.preprocessing import Imputer  # importing class for data missing values
imputer = Imputer(missing_values = 'NaN' , strategy = 'mean' , axis = 0)   # making object of that class
imp_1 = imputer.fit(DT_X[:, 0:2])  # fit above imputer object to our matrix X (we can do this way as well to select column by index no by  imp_1 = imputer.fit(X[:, [1,2]])   )
DT_X[:,  0:2] = imp_1.transform(DT_X[:,  0:2])   # LHS X dataset and RHs imp_object to implement missing values  (   X[:, 1:3] = imp_1.transform(X[:, 1:3])   )



# 4) Now we are going to start backward elimination and check covariance martrix
import statsmodels.formula.api as sm
# we are going to make a varibale of collection of independent/predictors
regressor_OLS = sm.OLS(endog = DT_Y , exog = DT_X).fit()  #Fit full model with all predictors
regressor_OLS.summary() # Check summary of model and remove highest P values predictors


# Covariance Matrix (0 means both variables are completly independent)
data_corr= pd.read_csv('Position_Salaries.csv')
cm = data_corr.corr()
sns.heatmap(cm,square = True)
plt.yticks(rotation = 0 )
plt.xticks(rotation = 90)
plt.show()


# 5) Random Forest Regression Model Application
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 500 , random_state=0)
regressor.fit(DT_X,DT_Y)


full_data = regressor.predict(DT_X) # Run model on full data
chk_data = regressor.predict(6.5)

# 6) Visulizing Data 
plt.scatter(DT_X,DT_Y,color = 'red')
plt.plot(DT_X,regressor.predict(DT_X), color = 'blue')
plt.title('Random Forest Model')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


##Result manulpulatrion
count = 0
vec = []
for i in full_data:
    if i == DT_Y[count]:
        print('same') 
        vec.append('Same')
    else:
        print('change' )
        vec.append('Change')
count = count + 1

#Dataframe conversion
df2 = pd.DataFrame(vec)
df1 = pd.DataFrame(full_data)
df0 = pd.DataFrame(DT_Y)
#Concat Result
result = pd.concat([df0,df1,df2], axis=1)

# Now check model on other dummy data
DT_dataset_new = pd.read_csv('Position_Salaries - pred.csv')
DT = DT_dataset_new.iloc[:,1:2].values
DC_Pred = regressor.predict(DT) 


# 6.1) Visulizing Data - Better way to scale chart
x_grid = np.arange(min(DT_X),max(DT_X),0.01)
x_grid = x_grid.reshape(len(x_grid),1)
 
plt.scatter(DT_X,DT_Y,color = 'red')
plt.plot(x_grid,regressor.predict(x_grid), color = 'blue')
plt.title('Random Forest Model')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()



