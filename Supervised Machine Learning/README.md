# Supervised Machine Learning with Iris Dataset: Naive Bayes, SVM, and KNN  

This project demonstrates the implementation of three popular supervised machine learning algorithms—Naive Bayes, Support Vector Machine (SVM), and K-Nearest Neighbors (KNN)—on the Iris Dataset. The notebook covers data preprocessing, training models, evaluating performance, and visualizing results.  

# Dataset  
The Iris dataset is a well-known dataset in machine learning, consisting of 150 samples of iris flowers. Each sample has four features:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width
  
The goal is to classify each sample into one of the three iris species:

- Iris-setosa
- Iris-versicolor
- Iris-virginica

# Project Structure
- **Data Preprocessing:**

  - Download the Iris dataset and assign proper column names.
  - Split the dataset into training and test sets (70% training, 30% testing).
  - Standardize the data using StandardScaler.

- **Naive Bayes Classification:**

  - Train a Naive Bayes classifier using the GaussianNB class from sklearn.
  - Evaluate training and test accuracy.
  - Plot confusion matrix and generate classification reports.
```python
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
sns.heatmap(result,
            annot = True,
            fmt = 'g',
            xticklabels = ['Setosa', 'Versicolor', 'Virginica'],
            yticklabels = ['Setosa', 'Versicolor', 'Virginica'])
plt.ylabel('Prediction', fontsize = 13)
plt.xlabel('Actual', fontsize = 13)
plt.title('Confusion Matrix', fontsize = 17)
plt.show()
```
![image](https://github.com/user-attachments/assets/41b87a19-34dc-4461-93b0-1c51b2563edc)

- **Support Vector Machine (SVM):**

  - Train SVM classifiers with different kernels: Linear, Polynomial, Radial Basis Function (RBF), and Sigmoid.
  - Compare accuracy for different kernels on both training and test datasets.
  - Plot accuracy comparison and visualize confusion matrices.
```python
# Import support vector classifier
# "Support Vector Classifier"

from sklearn.svm import SVC
accuracy_list = []

for i in ['linear', 'poly', 'rbf', 'sigmoid']:
  clf = SVC(kernel = i)
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_train)
  accuracy = accuracy_score(y_train, y_pred)
  accuracy_list.append(accuracy)

accuracies = {
    'Linear': accuracy_list[0],
    'Polynomial': accuracy_list[1],
    'RBF': accuracy_list[2],
    "Sigmoid": accuracy_list[3]
}
plt.figure(figsize=(8, 6))
sns.barplot(x = list(accuracies.keys()), y = list(accuracies.values()))
plt.xlabel('Kernel')
plt.ylabel('Accuracy')
plt.title('Accuracy of Different SVM Kernels on Iris Dataset')

# We have set y-axis limit for better visualization
plt.ylim(0.8, 1.0)
plt.show()
```
![image](https://github.com/user-attachments/assets/6cc6b6a2-76bb-43d1-bd49-a0d85277ed7a)

- **K-Nearest Neighbors (KNN):**

  - Train KNN classifiers for different values of K.
  - Find the optimal K by testing accuracies on both training and test datasets.
  - Visualize confusion matrices and print classification reports.
```python
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=10)
classifier.fit(X_train, y_train)
```
# Results  

- **Naive Bayes achieved:**
  - Training Accuracy: 97.14%
  - Test Accuracy: 93.73%
    
- **SVM achieved:**
  - Best Test Accuracy: 97.78% (Linear kernel)

- **KNN achieved:**
  - Best Test Accuracy: 97.78% at K = 10

# Libraries Used
- numpy
- pandas
- matplotlib
- seaborn
- sklearn

# Visualizations

- Confusion matrices for each classifier.
- Bar charts comparing SVM kernel accuracies and KNN accuracy for different K values.

# Conclusion

This project demonstrates the effectiveness of Naive Bayes, SVM, and KNN on a simple dataset like Iris. SVM with a linear kernel and KNN with K=10 provided the highest accuracy for this classification task.
