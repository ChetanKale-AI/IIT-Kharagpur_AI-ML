# K-Means Clustering using `sklearn` with Housing Dataset

This project demonstrates the use of the K-Means clustering algorithm using the `sklearn` package on a California housing dataset. The key steps include loading the dataset, feature extraction using PCA, performing clustering, and evaluating the clustering results using various metrics.

## Project Steps

**Step 1: Import necessary libraries and mount Google Drive**  
```python
from google.colab import drive
import pandas as pd
```
**Mount Google Drive**  
```python
drive.mount('/content/drive')
```
**Load dataset from Google Drive**  
```python
X = pd.read_csv("drive/My Drive/housing.csv")
print(X.head())
```
**Step 2: Select features for clustering** 
```python
data = list(zip(X.MedInc, X.HouseAge, X.AveRooms))
```
**Step 3: K-Means Clustering and Elbow Method to find the optimal number of clusters**  
```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
```
**List to store inertia for different numbers of clusters**  
```python
inertias = []
```
**Loop over a range of k (1 to 10)**  
```python
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, n_init='auto')
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)
```
**Plot the elbow method chart**  
```python
plt.plot(range(1, 11), inertias, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()
```
![Elbow Method](https://github.com/user-attachments/assets/1b32a7c4-fc1f-4804-a4cc-6bb9a84d219f)

**Step 4: Dimensionality Reduction using PCA**  
```python
from sklearn.decomposition import PCA
```
**Normalize the dataset**  
```python
X_norm = (X - X.min()) / (X.max() - X.min())
```
**Apply PCA to reduce dimensionality to 2D**  
```python
pca = PCA(n_components=2)
transformed = pd.DataFrame(pca.fit_transform(X_norm))
```
**Visualize the transformed data in 2D**  
```python
x = transformed[0]
y = transformed[1]

plt.scatter(x, y, color='black')
plt.title('PCA Reduced Data')
plt.show()
```
![image](https://github.com/user-attachments/assets/c3f1f1bb-a8ad-4f20-be59-04f088fe989a)

**Step 5: Apply K-Means Clustering with 2 clusters (based on Elbow method)**  
```python
kmeans = KMeans(n_clusters=2, n_init='auto')
kmeans.fit(data)
```
**Get the cluster labels**  
```python
c = kmeans.labels_
print(c)
```
**Visualize the clusters**  
```python
plt.scatter(x, y, c=kmeans.labels_)
plt.title('K-Means Clustering Visualization')
plt.show()
```
![image](https://github.com/user-attachments/assets/286a024a-97e6-4a4e-b4d8-2025d946939b)

**Step 6: Cluster Evaluation without Ground Truth**  

**Silhouette Coefficient**
```python
from sklearn import metrics

silhouette_score = metrics.silhouette_score(data, c, metric='euclidean')
print(f"Silhouette Coefficient: {silhouette_score}")
```
**Calinski-Harabasz Index**  
```python
calinski_harabasz_score = metrics.calinski_harabasz_score(data, c)
print(f"Calinski-Harabasz Index: {calinski_harabasz_score}")
```
**Davies-Bouldin Index**  
```python
davies_bouldin_score = metrics.davies_bouldin_score(data, c)
print(f"Davies-Bouldin Index: {davies_bouldin_score}")
```
# Outputs  
**Silhouette Coefficient: 0.559**  
**Calinski-Harabasz Index: 40641.45**  
**Davies-Bouldin Index: 0.604**  


