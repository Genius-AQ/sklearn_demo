"""
Reference: https://machinelearningcoban.com/2017/01/08/knn
"""
import numpy as np
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

"""
Load data
"""
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
print('Number of classes:', len(np.unique(iris_y)))
print('Number of data points:', len(iris_y))

# X0 = iris_X[iris_y == 0, :]
# X1 = iris_X[iris_y == 1, :]
# x2 = iris_X[iris_y == 2, :]

"""
Split data
"""
data_train, data_test, label_train, label_test = train_test_split(iris_X, iris_y, test_size=50)

"""
Create classifier & prediction
"""
clf = neighbors.KNeighborsClassifier(n_neighbors=5, p=2, weights='distance')
clf.fit(data_train, label_train)
y_pred = clf.predict(data_test)
# print('Predicted labels:', y_pred)
# print('Ground truth:    ', label_test)
# print('Accuracy:', 100*accuracy_score(label_test, y_pred), '%')
