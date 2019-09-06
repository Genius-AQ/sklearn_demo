"""
Reference: https://www.dataquest.io/blog/k-nearest-neighbors-in-python/
"""
import pandas
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

"""
Load data
"""
with open('iris.csv', 'r') as f:
    iris = pandas.read_csv(f)

# Attribute columns
X = iris.drop(columns=['class'])
# Class column
Y = iris['class'].values

"""
Split data
"""
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=50)

"""
Build a kNN classifier
"""
knn = KNeighborsClassifier(n_neighbors=5, p=2, weights='distance')
knn.fit(X_train, y_train)
prediction = knn.predict(X_test)

"""
Evaluate
"""
# mse = (((prediction - Y_test)**2).sum())/len(prediction)
# print('MSE:', mse)
print(100*knn.score(X_test, y_test), '%')
