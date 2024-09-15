# K-Means Clustering on Wine Dataset

This project demonstrates the application of K-Means Clustering on the wine dataset using Python and the scikit-learn package. The dataset contains various chemical properties of wine and is classified into three different wine categories. We use K-Means Clustering to group the wine samples and evaluate the clustering performance against the ground truth labels.  

# Dataset  
The dataset consists of 178 samples of wine with the following attributes:  

- Class: Type of wine (1, 2, or 3)  
- Chemical properties: Including Alcohol, Malic Acid, Ash, Magnesium, and others (13 features in total)  

## Project Steps

**Step 1: Data Loading and Preprocessing**  
```python
from google.colab import drive

drive.mount('/content/drive')

import pandas as pd

cols =  ['Class', 'Alcohol', 'MalicAcid', 'Ash', 'AlcalinityOfAsh', 'Magnesium', 'TotalPhenols',
         'Flavanoids', 'NonflavanoidPhenols', 'Proanthocyanins', 'ColorIntensity',
         'Hue', 'OD280/OD315', 'Proline']

D = pd.read_csv("drive/My Drive/wine.csv", names = cols)
D.head()
```
**Step 2: PCA for Dimensionality Reduction**  
We applied Principal Component Analysis (PCA) to reduce the dataset to two dimensions for visualization purposes.  

```python
from sklearn.decomposition import PCA

X_norm = (X - X.min())/(X.max() - X.min())

pca = PCA(n_components = 2)   #2 Dimensional PCA

transformed = pd.DataFrame(pca.fit_transform(X_norm))
```
**Step 3: Visualization**  
We visualized the transformed 2D data using a scatter plot.  
```python
import matplotlib.pyplot as plt

x = transformed[0]
y = transformed[1]

plt.scatter(x, y, color = 'black')
plt.show()
```
![image](https://github.com/user-attachments/assets/92d7a55b-d04c-4f94-af06-a5df5f5a1d7a)

**Elbow Method for Optimal K**  
To determine the optimal number of clusters, we used the Elbow Method by plotting the inertia for different values of k (number of clusters).  

```python
from sklearn.cluster import KMeans

data = list(zip(x, y))
inertias = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, n_init = 'auto')
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

plt.plot(range(1, 11), inertias, marker = 'o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()
```
![image](https://github.com/user-attachments/assets/769e701e-b5e2-4d8e-97a2-8f2d1f264e4f)

**Step 4: K-Means Clustering**  
We applied K-Means Clustering with 3 clusters, as the elbow plot suggested.  
```python
from sklearn.cluster import KMeans

data = list(zip(x, y))

kmeans = KMeans(n_clusters = 3, n_init = 'auto')
kmeans.fit(data)

c = kmeans.labels_
print(c)
```

**Step 5: Cluster Visualization**  
We plotted the clustered data using the K-Means labels.  

```python
plt.scatter(x, y, c = c)
plt.show()
```
![image](https://github.com/user-attachments/assets/c60f4e30-97f4-459f-a3ce-7a47957d5969)

**Step 6: Cluster evaluation with Ground Truth**  

**Rand Index**  
Given the knowledge of the ground truth class assignments labels_true and our clustering algorithm assignments of the same samples labels_pred, the (adjusted or unadjusted) Rand index is a function that measures the similarity of the two assignments, ignoring permutations:  
```python
from sklearn import metrics
from sklearn import metrics

labels_true = Y
labels_pred = c

metrics.rand_score(Y, c)
```
**Adjusted Rand Index**  
The Adjusted Rand Index corrects for chance in the Rand Index.  
```python
metrics.adjusted_rand_score(labels_true, labels_pred)
```
**Mutual Information based scores**  
AMI measures the agreement of the clustering labels with the ground truth, ignoring permutations.  
```python
metrics.adjusted_mutual_info_score(labels_true, labels_pred)
```
**Homogeneity, Completeness, and V-Measure**  
- Homogeneity: Each cluster contains only members of a single class.  
- Completeness: All members of a given class are assigned to the same cluster.  
- Their harmonic mean called V-measure is computed by v_measure_score.  

```python
metrics.homogeneity_score(labels_true, labels_pred)
metrics.completeness_score(labels_true, labels_pred)
metrics.v_measure_score(labels_true, labels_pred)
```
**Fowlkes-Mallows Index (FMI)**  
The FMI is the geometric mean of precision and recall for clustering.  
```python
metrics.fowlkes_mallows_score(labels_true, labels_pred)
```

# Outputs
**Rand Index: 0.9318**  
**Adjusted Rand Index: 0.8471**  
**AMI: 0.8329**  
**Homogeneity: 0.8375**  
**Completeness: 0.8319**  
**V-Measure: 0.8347**  
**FMI: 0.8984**  

# Conclusion
This project demonstrates how to implement K-Means Clustering on a real-world dataset, evaluate it using various clustering metrics, and visualize the results. Our clustering model performs well when compared to the ground truth with a high Rand Index and Adjusted Mutual Information score.  
