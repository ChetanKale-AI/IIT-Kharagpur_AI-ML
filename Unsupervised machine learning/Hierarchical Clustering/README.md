# Hierarchical Clustering using Scikit-learn and Scipy  

**Overview**  
This project demonstrates how to implement Hierarchical Clustering using the scipy and sklearn packages in Python. It walks through the process of computing proximity matrices, performing agglomerative clustering using various linkage methods, and visualizing the resulting dendrograms. Additionally, post-clustering evaluation metrics such as the Silhouette Coefficient, Calinski-Harabasz Index, and Davies-Bouldin Index are computed to assess clustering quality.

**Import necessary libraries**  
```python
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn import metrics
```
**Step 1: Visualize and Understand the Input Data**  
```python
x = [ 5,  4, 13, 11, 13,  5,  3,  4,  3, 10, 12, 13, 14, 12, 11,  6,  7,  6,  5, 11, 14, 10, 12, 13, 14, 12, 11, 13, 11, 13, 14, 15]
y = [15, 18, 22, 21, 21, 19, 15, 16, 16, 21, 23, 22, 22, 20, 22, 17, 18, 14, 17, 24, 22, 23,  1,  2,  4,  1,  3,  6,  3,  2,  6,  4]

plt.scatter(x, y, color = 'black')
plt.show()
```

![image](https://github.com/user-attachments/assets/5520fdf5-dd37-4aa7-927c-23073bb6357f)

**Step 2: Perform Hierarchical Clustering using various linkage methods**  
```python
data = list(zip(x, y))

# Linkage methods
Z1 = linkage(data, method='single', metric='euclidean')    # Single linkage (Min)
Z2 = linkage(data, method='complete', metric='euclidean')  # Complete linkage (Max)
Z3 = linkage(data, method='average', metric='euclidean')   # Average linkage
Z4 = linkage(data, method='ward', metric='euclidean')      # Ward linkage
Z5 = linkage(data, method='centroid', metric='euclidean')  # Centroid linkage
```
**Step 3: Visualize Dendrograms for each linkage method**  

**Single linkage dendrogram**  
```python
# Single linkage dendogram plot
plt.plot(2,2,1), dendrogram(Z1), plt.title('Single')
plt.show()
```
![image](https://github.com/user-attachments/assets/6f539c10-e76a-4d33-86e7-5242b1de3a9a)

**Complete linkage dendrogram**  
```python
# Complete linkage dendogram plot
plt.plot(2,2,2), dendrogram(Z2), plt.title('Complete')
plt.show()
```
![image](https://github.com/user-attachments/assets/d0e9f7fb-bc54-4ac6-8bc9-127ca9b10837)

**Average linkage dendrogram**  
```python
# Average linkage dendogram plot
plt.plot(2,2,3), dendrogram(Z3), plt.title('Average')
plt.show()
```
![image](https://github.com/user-attachments/assets/e0be86ab-5d1f-4600-8787-a57cd161731b)

**Ward linkage dendrogram**  
```python
# Ward linkage dendogram plot
plt.plot(2,2,4), dendrogram(Z4), plt.title('Ward')
plt.show()
```
![image](https://github.com/user-attachments/assets/e6fca371-21d1-458e-88dd-9512ee65cb0a)

**Centroid linkage dendrogram**  
```python
# Centroid linkage dendogram plot
plt.plot(2,2,5), dendrogram(Z5), plt.title('Centroid')
plt.show()
```
![image](https://github.com/user-attachments/assets/e82c84b1-997d-42fa-8b38-264073e91a59)


**Step 4: Post-Pruning to Get a Specific Number of Clusters (for example, 3 clusters from Ward linkage)**  
```python
from scipy.cluster.hierarchy import fcluster

c = fcluster(Z4, 3, criterion = 'maxclust')

print(f"Clusters: {c}")
```

**Step 5: Cluster Evaluation Without Ground Truth**  

# Silhouette Coefficient
```python
from sklearn import metrics

metrics.silhouette_score(data, c, metric = 'euclidean')
```
**Silhouette Coefficient: 0.7508**  

# Calinski-Harabasz Index
```python
metrics.calinski_harabasz_score(data, c)
```
**Calinski-Harabasz Index: 291.0977**  

# Davies-Bouldin Index
```python
metrics.davies_bouldin_score(data, c)
```
**Davies-Bouldin Index: 0.3373**

# Conclusion:  
 The clustering evaluation metrics (Silhouette, Calinski-Harabasz, and Davies-Bouldin) help us understand the clustering performance.  
- A Silhouette score of 0.7508 suggests the clusters are dense and well-separated.  
- The Calinski-Harabasz score of 291.0977 indicates a strong definition of clusters.  
- A Davies-Bouldin score of 0.3373 further supports that the clusters are well-separated and compact.  
