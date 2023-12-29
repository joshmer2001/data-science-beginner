import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA 

raw_data = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv') 

#K-Means clustering
#raw_data['species'].unique()
data = raw_data.replace('setosa', 0).replace('versicolor', 1).replace('virginica', 2)



scaler=StandardScaler() #init of scaler to standardise/transform all various features into one scale
features_ = scaler.fit(data) #fit to data
features = features_.transform(data) #apply transformation to data
scaled = pd.DataFrame(features, columns = data.columns ) #question: did it need to be scaled if they are all in mm?
x = scaled.values

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

wcss = [] #Within Cluster Sum of Squares
for i in range(1,10):
    kmeans_model = KMeans(n_clusters=i, random_state=20)
    kmeans_model.fit(x)
    wcss.append(kmeans_model.inertia_)

#print(wcss)

plt.plot([x for x in range(1,10)], wcss, 'gs-') #green solid
plt.xlabel("Values of 'k'")
plt.ylabel('WCSS')
plt.plot([3], [wcss[2]], 'ro', ms=12, mfc='r')
#plt.show()

#elbow at 3 clusters

kmeans = KMeans(n_clusters=3, init='k-means++', random_state=20)
kmeans.fit(x)
centres = kmeans.cluster_centers_

x_ = scaled[['petal_length', 'petal_width']].values
li = 2
wi = 3

lc = centres[:,li]
wc = centres[:,wi]


fig, (p1, p2) = plt.subplots(1, 2, figsize = (14,5))

p1.scatter(x_[:,0], x_[:,1], c=kmeans.labels_)
p1.scatter(lc[:], wc[:], marker='x', s=100,c='red')
p1.set_xlabel('Petal Width')
p1.set_ylabel('Petal Length')
p1.set_title('Predicted Clusters')


p2.scatter(x_[:,0], x_[:,1], c=data['species'])
p2.set_xlabel('Petal Width')
p2.set_ylabel('Petal Length')
p2.set_title('Original Data')
plt.tight_layout()
plt.show()

from sklearn.metrics import silhouette_score
print (f'Silhouette score: {silhouette_score(x_, kmeans.labels_)}')