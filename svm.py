"""
Reference: https://note.nkmk.me/python-scikit-learn-svm-iris-dataset/
"""
import pandas as pd
from sklearn import datasets, model_selection, svm, metrics

"""
Load data
"""
iris = datasets.load_iris()
iris_data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_label = pd.Series(data=iris.target)

"""
Split data
"""
data_train, data_test, label_train, label_test = model_selection.train_test_split(iris_data, iris_label)

"""
Create classifier then predict
"""
clf = svm.SVC()
clf.fit(data_train, label_train)
pred = clf.predict(data_test)
accuracy = metrics.accuracy_score(label_test, pred)
print('Accuracy:', accuracy*100, '%')
