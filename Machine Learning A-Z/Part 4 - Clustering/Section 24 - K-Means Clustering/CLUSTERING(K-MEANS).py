# K mean Clustering
import numpy as np                 # Mathematics fucntopns
import matplotlib.pyplot as plt    # Plot and Charts
import pandas as pd                # Data Set management and import them
import seaborn as sns              # Covariance Matrix


################################################### DATA PREPROCESSING #################################
# 2) importing Dataset
DT_dataset = pd.read_csv('Mall_Customers.csv')
DT_X = DT_dataset.iloc[:,[3,4]].values   # dependend variable matrix means that will helo in prediction [rows , columns]

DT_dataset.shape

###In k mean we have decide number of cluster,
## we are going find of optimal no of cluster by using elbow method

from sklearn.cluster import KMeans
wcss = []   # WCSS formula is we applying....
for i in range(1 , 11):
    kmeans = KMeans(n_clusters = i , init = 'k-means++' , max_iter = 300, n_init = 10 , random_state = 0 )
    kmeans.fit(DT_X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('wcss')
plt.show()

##as we can see in plot 5 is the best no cluster we can choose through elbow method


#Apply keamsn to to mall dataset
kmeans = KMeans(n_clusters = 5 , init = 'k-means++' , max_iter = 300, n_init = 10 , random_state = 0 )
result = kmeans.fit_predict(DT_X)

a =pd.DataFrame(result)
a2  = np.append(DT_dataset, a, 1)
a3 =pd.DataFrame(a2)



## Visualize 
plt.scatter(DT_X[result ==0,0],DT_X[result ==0,1] , s = 100, c = 'red', label = 'cluster 1')
plt.scatter(DT_X[result ==1,0],DT_X[result ==1,1] , s = 100, c = 'blue', label = 'cluster 2')
plt.scatter(DT_X[result ==2,0],DT_X[result ==2,1] , s = 100, c = 'green', label = 'cluster 3')
plt.scatter(DT_X[result ==3,0],DT_X[result ==3,1] , s = 100, c = 'cyan', label = 'cluster 4')
plt.scatter(DT_X[result ==4,0],DT_X[result ==4,1] , s = 100, c = 'magenta', label = 'cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0],kmeans.cluster_centers_[:, 1] , s = 300, c = 'yellow', label = 'Centroid')
plt.title('Clusters of Clients')
plt.xlabel('Annual income')
plt.ylabel('Spending Score [1-100]')
plt.legends()
plt.show()
