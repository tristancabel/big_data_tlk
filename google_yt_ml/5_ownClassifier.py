import random
import numpy as np
from sklearn.datasets import load_iris
import  sklearn.tree as tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from scipy.spatial import distance

class RandomClf():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train =y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = random.choice(self.y_train)
            predictions.append(label)
        return predictions

class ScrappyKNN():
    def euc(self, a, b):
        return distance.euclidean(a,b)

    def closest(self, row):
        best_dist = self.euc(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = self.euc(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train =y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions





def printAccuracy(clf, X, y, test_sz):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_sz)
    clf.fit(X_train, y_train)
    my_pred = clf.predict(X_test)
    print "accuracy %s %0.2f" % (clf.__class__.__name__, accuracy_score(y_test, my_pred))


iris = load_iris()
X = iris.data
y = iris.target

## decision tree
printAccuracy(tree.DecisionTreeClassifier(), X, y, 0.5)

## K-neighbors
printAccuracy(KNeighborsClassifier(), X, y, 0.5)

## srandomClf
printAccuracy(RandomClf(), X, y, 0.5)

## scrappy KNN
printAccuracy(ScrappyKNN(), X, y, 0.5)
