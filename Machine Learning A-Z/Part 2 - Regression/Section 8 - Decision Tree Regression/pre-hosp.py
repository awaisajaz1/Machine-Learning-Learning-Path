# Decision Tree Regression

import numpy as np                 # Mathematics fucntopns
import matplotlib.pyplot as plt    # Plot and Charts
import pandas as pd                #Data Set management and import them
import seaborn as sns              # Covariance Matrix


# 2) importing Dataset
DT_dataset = pd.read_csv('pre2.csv')
DT_X = DT_dataset.iloc[:,0:9].values   # dependend variable matrix means that will helo in prediction [rows , columns]
DT_Y = DT_dataset.iloc[:,9].values      # independent variable vector means that need to be predicted [rows , columns]


# Categorical Data and Encoding it - in our case STATE is categorical variable
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X_1 = LabelEncoder()
DT_X[:, 0]  = labelencoder_X_1.fit_transform(DT_X[:, 0])

labelencoder_X_2 = LabelEncoder()
DT_X[:, 8]  = labelencoder_X_2.fit_transform(DT_X[:, 8])


labelencoder_Y_1 = LabelEncoder()
DT_Y = labelencoder_Y_1.fit_transform(DT_Y)


#  Split Data into Train and Test
from sklearn.model_selection import train_test_split
X_train , X_test , Y_train, Y_test = train_test_split(DT_X,DT_Y , test_size = 0.40 , random_state=0) #test_size can be 0.3 or 1/3



# 5) Decsion Tree Regression Model Application
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train,Y_train)
pred_y = regressor.predict(X_test)


# ) Fitting Naive Bayes Classifier to Training Set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,Y_train)
 #Predicting the Test Result
svm_pred = classifier.predict(X_test)


##Accuracy Score
dt = 0
for i in range(len(pred_y)):
    if pred_y[i] == Y_test[i]:  # by array pointer we are looping into values to see same value if found then we increment oy count  variable
        dt += 1
        
svm = 0
for i in range(len(svm_pred)):
    if svm_pred[i] == Y_test[i]:
        svm += 1
        

print("Decison Tree Accuracy: %.2f%%" % ((dt/float(len(Y_test)) * 100)))
print("Support Vector Machine Accuracy: %.2f%%" % ((svm/float(len(Y_test)) * 100)))


dt1 = (dt/float(len(Y_test)) * 100)
svm1 = (svm/float(len(Y_test)) * 100)

if  dt1 > svm1:
    from sklearn.metrics import confusion_matrix
    print("Decision tree has more accuracy , implemening please wait")
    full_data = regressor.predict(DT_X) # Run model on full data
    cm2 = confusion_matrix(DT_Y, full_data)
    print(cm2)
else:
    from sklearn.metrics import confusion_matrix
    print("Support Vector Machine has more accuracy , implemening please wait")
    full_data = classifier.predict(DT_X) # Run model on full data
    cm2 = confusion_matrix(DT_Y, full_data)
    print(cm2)
    

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, pred_y)





full_data = regressor.predict(DT_X) # Run model on full data
cm2 = confusion_matrix(DT_Y, full_data)



 


