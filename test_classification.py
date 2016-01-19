import drzewo_dodana_regresja
import numpy
from sklearn import datasets
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from mapowanie import *

iris = datasets.load_iris()
X_easy = iris.data[:100,:]
print "X_easy", X_easy
y_easy = iris.target[:100]
print "y_easy", y_easy
y_i = mapping(y_easy)
print "y_i", y_i
X_hard = iris.data[50:,:]
y_hard = iris.target[50:]
y_h = mapping(y_hard)

# gives a percentage of correct predictions in 10-fold crossvalidation
"""def get_crossvalidated_score(r, X, y):

    kf = cross_validation.KFold(n=len(X), n_folds=10)
    all_cases = 0
    correct = 0

    def no_correct(preds, truth):
        return sum(map(lambda (x, y): x == y, zip(preds, truth)))
        
    for train, test in kf:
        print "y_train", y
        r.fit(X[train], y[train])
        correct += no_correct(r.predict(X[test]), y[test])
        all_cases += len(test)
    return float(correct) / all_cases"""

def get_score(truth, predictions):
    return sum(map(lambda (x, y): 1 if x == y else 0, zip(truth, predictions)))

#sklearn_classifier = RandomForestClassifier()
#users_classifier = drzewo_dodana_regresja.RandomForestClassifier(3)

# easy case
#print "Iris setosa vs iris versicolor - sklearn: %f" % get_crossvalidated_score(sklearn_classifier, X_easy, y_easy)
#print "Iris setosa vs iris versicolor - user's code: %f" % get_crossvalidated_score(users_classifier, X_easy, y_i)
#users_classifier.fit(X_easy,y_i)
#users_classifier.predict(X_easy)
# harder cas
#print "Iris versicolor vs iris virginica - sklearn: %f" % get_crossvalidated_score(sklearn_classifier, X_hard, y_hard)
#print "Iris versicolor vs iris virginica - user's code: %f" % get_crossvalidated_score(users_classifier, X_hard, y_h)

classifier = drzewo_dodana_regresja.RandomForestClassifier(3)
classifier.fit(X_hard,y_h)
predictions = classifier.predict(X_hard)
predictions_proba = classifier.predict_proba(X_hard)
print "predictions_proba", predictions_proba
print "custom: %f" % get_score(y_h, predictions)