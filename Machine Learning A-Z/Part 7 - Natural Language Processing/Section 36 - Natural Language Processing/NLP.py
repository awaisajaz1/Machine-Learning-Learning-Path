# NLP
#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset as tsv
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter= '\t', quoting = 3)
#dataset[['Review','Liked']]
# Cleaning the texts
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

row = dataset.shape[0]
corpus = []
for i in range(0 , row):
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review = review.lower()
    review = review.split() # word tokenization
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    


# Creating the Bag of Words model term document matrix
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()


cv.get_feature_names()
a1 = cv.inverse_transform(X)
y = dataset.iloc[:,1].values



# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Fitting Decision tree to the Training set
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train, y_train)


# Fitting Random Forest to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier2 = RandomForestClassifier(n_estimators = 300 ,criterion = 'entropy', random_state=0)
classifier2.fit(X_train,y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y2_pred = regressor.predict(X_test)
y3_pred = classifier2.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm2 = confusion_matrix(y_test, y2_pred)

################################################################testing ========================
from nltk import sent_tokenize, word_tokenize, pos_tag
test = 'i is am happy. He is bad. Wow its a great thing'
corpus2 = []
test = sent_tokenize(test)
for inp in test:
    inp= re.sub('[^a-zA-Z]',' ',inp)
    inp= inp.lower()
    inp = inp.split()
    ps = PorterStemmer()
    stopwords = nltk.corpus.stopwords.words("english")
    test= [ps.stem(word) for word in inp if not word in set(stopwords)]
    inp = ' '.join(test)
    corpus2.append(inp)

a2 = cv.transform(corpus2).toarray()
answer= classifier.predict(a2)
answer2 = regressor.predict(a2)
anser3 = classifier2.predict(a2)


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 5)
accuracies
accuracies.mean()
accuracies.std()
