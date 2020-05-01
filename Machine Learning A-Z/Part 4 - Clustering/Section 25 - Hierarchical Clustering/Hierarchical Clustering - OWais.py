#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import mall dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

#using dendogram to find iptimal no of clusters
import scipy.cluster.hierarchy as sch  # dendogram library
#method = ward    minimumize the variance within each cluster
dendogram = sch.dendrogram(sch.linkage(X,method = 'ward'))
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()



##Fitting hierarchical cuter to ou mall dataset

#Egglomerative (bottom to top)
# n_clusters =5 is number of clusters we get from dendogram method
# affinity = 'euclidean' we choose points betwen clusters
# Method linkage = 'ward'  minimumize the variance within each cluster
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5 , affinity = 'euclidean' , linkage = 'ward')
y_pred = hc.fit_predict(X)


#Visulaize the result
# Visualising the clusters
plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], s = 100, c = 'red', label = 'Shiekhlog')
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], s = 100, c = 'blue', label = '#standard')
plt.scatter(X[y_pred == 2, 0], X[y_pred == 2, 1], s = 100, c = 'green', label = 'Ameerlog')
plt.scatter(X[y_pred == 3, 0], X[y_pred == 3, 1], s = 100, c = 'cyan', label = 'Bachaywithabbamoney')
plt.scatter(X[y_pred == 4, 0], X[y_pred == 4, 1], s = 100, c = 'magenta', label = 'Maturelog')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()