
#Churn analysis of bank leaving customer and we will calssify them
# Artificial Neural Network

# Installing Theano (Numerical based library (runs on cpu aswellas gpu))
# !pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow ()
# !Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras ()
# !pip install  keras
# !pip install --upgrade theano
# !pip install pandas
# !pip install matplotlib
# !pip install seaborn 
# !pip install sklearn 
# !pip install pydot

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import theano as th
import tensorflow as tf
import seaborn as sns  

#hello = tf.constant('Hello, TensorFlow!')
#sess = tf.Session()
#print(sess.run(hello))

######################################################## Part 1 - Data Preprocessing

# importing dataset
dataset = pd.read_csv("Churn_Modelling.csv")
DT_X = dataset.iloc[:,3:13].values
DT_Y = dataset.iloc[:,13].values


# Categorical Data and Encoding it - in our case STATE is categorical variable
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelencoder_X_1 = LabelEncoder()
DT_X[:, 1]  = labelencoder_X_1.fit_transform(DT_X[:, 1])

labelencoder_X_2 = LabelEncoder()
DT_X[:, 2]  = labelencoder_X_2.fit_transform(DT_X[:, 2])
#X = np.array(X,dtype= 'int64')
#Here we are splitting categorial variable states in column as no of catogories - no of columns
onehotencoder = OneHotEncoder(categorical_features = [1])
DT_X = onehotencoder.fit_transform(DT_X).toarray() # Dummy variable  is created against state
DT_X = DT_X[:, 1:]


#  Split Data into Train and Test
from sklearn.model_selection import train_test_split
X_train , X_test , Y_train, Y_test = train_test_split(DT_X,DT_Y , test_size = 0.20 , random_state=0) #test_size can be 0.3 or 1/3


# Feature Scaling
from sklearn.preprocessing import StandardScaler
std_X = StandardScaler()
X_train= std_X.fit_transform(X_train)
X_test = std_X.transform(X_test)

full_x  = DT_X
full_x = std_X.fit_transform(full_x)


######################################################## Part 2  - Making Artificial Neural Network

# 1) Import Keras libraries and its modules
import keras 
from keras.models import Sequential   # to initialize neural network
from keras.layers import Dense    # use to create layers in ann
from keras.utils.vis_utils import plot_model
#import pydot
# Initialize Artificial Neural Network (We will generate sequence of layers)
#Activation function we will work are  1) Reactifier Aactivitation functiona and sigmoid
classifier = Sequential() 
## We are choosing retifier activation function for hidden layers and segmoid for output layers

# Initialize the input layer and first hidden layer
#Dense is going to use initilize weights
#Add method is used to add layers 
# 6 = number of nodes in hidden layer(how to determine numbers of layer to be added) so we will do as (11 independent var + 1 dependent var = 12 so we will take average = 12/2 = 6)
# input_dim = no of independent variables 
# kernel_initializer = weights to be uniform
# activation = activation functionof hidden layer
# units = number of neurons(perceptrons)
classifier.add(Dense(units = 6, input_dim=11, kernel_initializer ='uniform', activation='relu'))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the 3rd hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the 4rd hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
#Adding the output layer
# units = units = number of neurons(perceptrons)
# kernel_initializer = 'uniform'  this function will initialize weight from its back hidden layer so in this case from 3rd layer
# kernel_initializer = 'uniform'  it will also initialize weight v near to zero
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))


# Compiling the ANN (stochastic gradient descent)
# optimizer = will determine the best weights
# metrics= to evualte the model (after backpropogation this function evulate the model and improve the model performce)
# loss=binary_crossentropy it helps optimer  = "adams" to find optimal weight (for categorical dependent variable loss = categorical_crossentropy)
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Fitting the ANN to the Training set
# nb_epoch = number of times to train ann whole training set and we will see its improvement at every nb_epoch means steps repeatition
# batch_size = number of observations check  after weight will update
classifier.fit(X_train, Y_train, batch_size = 10, nb_epoch = 20)

print(classifier.summary())
plot_model(classifier, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
# Part 3 - Making the predictions and evaluating the model
# Predicting the Test set results
y_pred = classifier.predict(X_test) #it is Y_test prediction


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
y_pred2 = (y_pred > 0.5)   # For confusion matrix we change the result into true false other wise it sill not work on probability
cm = confusion_matrix(Y_test, y_pred2)


# Now run on full data
YFULL_pred = classifier.predict(full_x)
YFULL_pred = (YFULL_pred > 0.5)   # For confusion matrix we change the result into true false other wise it sill not work on probability
cm2 = confusion_matrix(DT_Y , YFULL_pred)



from keras.utils import plot_model
plot_model(classifier, to_file='model.png')