import drzewo_dodana_regresja
import numpy
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

iris = datasets.load_iris()
X = iris.data[:100,:]
y = iris.target[:100]
print X
print y

def get_score(truth, predictions):
    return sum(map(lambda (x, y): 1 if x == y else 0, zip(truth, predictions)))

sklearn_classifier = RandomForestClassifier()
sklearn_classifier = sklearn_classifier.fit(X, y)
predictions = sklearn_classifier.predict(X)
print "sklearn: %f" % get_score(y, predictions)

classifier = drzewo_dodana_regresja.RandomForestClassifier(n_features_user=4)
classifier.fit(X,y)
predictions = classifier.predict(X)
print "custom: %f" % get_score(y, predictions)
