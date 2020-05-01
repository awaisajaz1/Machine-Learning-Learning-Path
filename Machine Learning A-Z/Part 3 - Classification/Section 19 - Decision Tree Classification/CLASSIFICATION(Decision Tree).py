
# Decision Tree Classification


import numpy as np                 # Mathematics fucntopns
import matplotlib.pyplot as plt    # Plot and Charts
import pandas as pd                # Data Set management and import them
import seaborn as sns              # Covariance Matrix
from pandas.plotting import scatter_matrix


################################################### DATA PREPROCESSING #################################
# 2) importing Dataset
DT_dataset = pd.read_csv('Social_Network_Ads.csv')
print(DT_dataset.shape)
# head
print(DT_dataset.head(5))
DT_X = DT_dataset.iloc[:,[1,2,3]].values   # dependend variable matrix means that will helo in prediction [rows , columns]
DT_Y = DT_dataset.iloc[:,4].values      # independent variable vector means that need to be predicted [rows , columns]


df= pd.DataFrame(y_pred_full , columns = ['result'])
        
        


# descriptions
print(DT_dataset.describe())

# class distribution
print(DT_dataset.groupby('Purchased').size())


# histograms
DT_dataset.hist()
plt.show()

# scatter plot matrix
scatter_matrix(DT_dataset)
plt.show()

# Rotate axis labels and remove axis ticks
axs = pd.scatter_matrix(DT_dataset, figsize=(4, 4))
n = len(DT_dataset.columns)
for i in range(n):
    v = axs[i, 0]
    v.yaxis.label.set_rotation(0)
    v.yaxis.label.set_ha('right')
    v.set_yticks(())
    h = axs[n-1, i]
    h.xaxis.label.set_rotation(90)
    h.set_xticks(())

#import pandas
#from pandas.plotting import scatter_matrix
#import matplotlib.pyplot as plt
#from sklearn import model_selection
#from sklearn.metrics import classification_report
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import accuracy_score
#from sklearn.linear_model import LogisticRegression
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.naive_bayes import GaussianNB
#from sklearn.svm import SVC
## Spot Check Algorithms
#
#
## 4) Categorical Data and Encoding it - in our case STATE is categorical variable
#from sklearn.preprocessing import LabelEncoder,OneHotEncoder
#labelencoder_X = LabelEncoder()
#DT_X[:, 0]  = labelencoder_X.fit_transform(DT_X[:, 0])
##X = np.array(X,dtype= 'int64')
##Here we are splitting categorial variable states in column as no of catogories - no of columns
#onehotencoder = OneHotEncoder(categorical_features = [3])
#DT_X = onehotencoder.fit_transform(DT_X).toarray() # Dummy variable  is created against state
#
#
#
#from sklearn.cross_validation import train_test_split
#X_train , X_test , Y_train, Y_test = train_test_split(DT_X,DT_Y , test_size = 0.3 , random_state=0) #test_size can be 0.3 or 1/3
#
#
#models = []
#models.append(('LR', LogisticRegression()))
#models.append(('LDA', LinearDiscriminantAnalysis()))
#models.append(('KNN', KNeighborsClassifier()))
#models.append(('CART', DecisionTreeClassifier()))
#models.append(('SVM', SVC()))
## evaluate each model in turn
#results = []
#names = []
#for name, model in models:
#	kfold = model_selection.KFold(n_splits=10, random_state=0)
#	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
#	results.append(cv_results)
#	names.append(name)
#	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
#	print(msg)
#    
#    
    
    
    
# 3)Taking Care of Missing Data (We take mean of that column and put in missing area thats simple)
from sklearn.preprocessing import Imputer  # importing class for data missing values
imputer = Imputer(missing_values = 'NaN' , strategy = 'mean' , axis = 0)   # making object of that class
imp_1 = imputer.fit(DT_X[:, 1:3])  # fit above imputer object to our matrix X (we can do this way as well to select column by index no by  imp_1 = imputer.fit(X[:, [1,2]])   )
DT_X[:,  1:3] = imp_1.transform(DT_X[:,  1:3])   # LHS X dataset and RHs imp_object to implement missing values  (   X[:, 1:3] = imp_1.transform(X[:, 1:3])   )



# 4) Categorical Data and Encoding it - in our case STATE is categorical variable
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
DT_X[:, 0]  = labelencoder_X.fit_transform(DT_X[:, 0])



#X = np.array(X,dtype= 'int64')
#Here we are splitting categorial variable states in column as no of catogories - no of columns
onehotencoder = OneHotEncoder(categorical_features = [0])
DT_X = onehotencoder.fit_transform(DT_X).toarray() # Dummy variable  is created against state


# 5) Now we are going to start backward elimination and see covariance matrix
#we are just making copy of of variableshere to test from zero

# Covariance Matrix (0 means both variables are completly independent)
data_corr= pd.read_csv('Social_Network_Ads.csv')
cm = data_corr.corr()
sns.heatmap(cm,square = True)
plt.yticks(rotation = 0 )
plt.xticks(rotation = 90)
plt.show()



X1 = DT_X
Y1 = DT_Y
import statsmodels.formula.api as sm
# we are going to make a varibale of collection of independent/predictors
# Confidence of Variable by looking P value
X1 = np.matrix(DT_X , dtype = 'float')
X1_optimized = X1  # We will delete index step by step to get important predictors
regressor_OLS = sm.OLS(endog = Y1, exog = X1_optimized).fit()  #Fit full model with all predictors
regressor_OLS.summary() 
# a larger (insignificant) p-value suggests that changes in the predictor are not associated with changes in the response.


# 6) Split Data into Train and Test
from sklearn.cross_validation import train_test_split
X_train , X_test , Y_train, Y_test = train_test_split(DT_X,DT_Y , test_size = 0.3 , random_state=0) #test_size can be 0.3 or 1/3



 # 7) Feature Scaling
from sklearn.preprocessing import StandardScaler
std_X = StandardScaler()
X_train= std_X.fit_transform(X_train)
X_test = std_X.transform(X_test)
DT_X = std_X.transform(DT_X)

################################################### Model Training and Results#######################################

# 8) Fitting DECISION TREE to Training Set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)


# 9) Predicting the Test Result
y_pred = classifier.predict(X_test)

y_pred_full = classifier.predict(DT_X) 


#confusion matrix to check Prdiction Results
from sklearn.metrics import confusion_matrix
con_m = confusion_matrix(Y_test,y_pred)
con_m2 = confusion_matrix(DT_Y,y_pred_full)




df= pd.DataFrame(y_pred_full , columns = ['result'])


  

################################################### Model Visulualization #################################

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, ].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('DECISION TREE (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


plt.savefig('figure.png')



# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.001),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.001))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('DECISION TREE Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()












