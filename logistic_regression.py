"""
Reference: https://slundberg.github.io/shap/notebooks/Iris%20classification%20with%20scikit-learn.html
"""
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.2)

"""
START HERE
"""
def print_accuracy(f):
    print("Accuracy = {0}%".format(100*np.sum(f(X_test) == y_test)/len(y_test)))


linear_lr = LogisticRegression()
linear_lr.fit(X_train, y_train)
print_accuracy(linear_lr.predict)
