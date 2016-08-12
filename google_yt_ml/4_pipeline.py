import numpy as np
from sklearn.datasets import load_iris
import  sklearn.tree as tree

iris = load_iris()

X = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

## decision tree
from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()

my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print "accuracy DecisionTree %0.2f" % accuracy_score(y_test, predictions)


## K-neighbors
from sklearn.neighbors import KNeighborsClassifier
knei = KNeighborsClassifier()

knei.fit(X_train, y_train)
knei_pred = knei.predict(X_test)
print "accuracy KNeighborsClassifier %0.2f" % accuracy_score(y_test, knei_pred)
