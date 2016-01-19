import randomforests
import numpy
from sklearn import datasets
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
#from mapowanie import *

def mapping(cos):
    a = cos[0]
    for i in range(len(cos)):
        if cos[i]==a:
            cos[i] = 0
        else:
            cos[i] = 1
    return cos

iris = datasets.load_iris()
X_easy = iris.data[:100,:]
y_easy = iris.target[:100]
X_hard = iris.data[50:,:]
y_hard = iris.target[50:]


# gives a percentage of correct predictions in 10-fold crossvalidation
def get_crossvalidated_score(r, X, y):

    kf = cross_validation.KFold(n=len(X), n_folds=10)
    all_cases = 0
    correct = 0

    def no_correct(preds, truth):
        return sum(map(lambda (x, y): x == y, zip(preds, truth)))
        
    for train, test in kf:
        r.fit(X[train], y[train])
        correct += no_correct(r.predict(X[test]), y[test])
        all_cases += len(test)
    return float(correct) / all_cases


sklearn_classifier = RandomForestClassifier()
users_classifier = randomforests.RandomForestClassifier(3)
y_i = mapping(y_easy)
# easy case
print "Iris setosa vs iris versicolor - sklearn: %f" % get_crossvalidated_score(sklearn_classifier, X_easy, y_easy)
print "Iris setosa vs iris versicolor - user's code: %f" % get_crossvalidated_score(users_classifier, X_easy, y_i)

y_h = mapping(y_hard)

# harder case
print "Iris versicolor vs iris virginica - sklearn: %f" % get_crossvalidated_score(sklearn_classifier, X_hard, y_hard)
print "Iris versicolor vs iris virginica - user's code: %f" % get_crossvalidated_score(users_classifier, X_hard, y_h)
