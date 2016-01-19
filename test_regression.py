import drzewo_dodana_regresja
import numpy, copy, random
from sklearn.ensemble import RandomForestRegressor as sklearnRandomForestRegressor
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn import cross_validation


f = open("forestfires_log.csv")
X = []
y = []
for l in f:
    l=l.split()
    X.append(l[:-1])
    y.append(l[-1])
z = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11]
for i in range(len(X)):
    for v in z:
        X[i][v] = float(X[i][v])
y = map(float, y)
le = preprocessing.LabelEncoder()
months = map(lambda x: x[2], X)
le.fit(months)
months_labels = le.transform(months)
days = map(lambda x: x[3], X)
le.fit(days)
days_labels = le.transform(days)
# scikit cannot deal with categorical variables...
X_mod = copy.deepcopy(X)
for i in range(len(X)):
    X_mod[i][2] = months_labels[i]
    X_mod[i][3] = days_labels[i]

tupled = zip(X, X_mod, y)
random.shuffle(tupled)
X, X_mod, y = zip(*tupled)
X = numpy.array(X)
X_mod = numpy.array(X_mod)
y = numpy.array(y)

def cross_val_scores(r, X, y, cv):
    kf = cross_validation.KFold(n=len(X), n_folds=cv)
    scores = []
    for train, test in kf:
        r.fit(X[train], y[train])
        scores.append(mean_squared_error(r.predict(X[test]), y[test]))
    return scores

r = RandomForestRegressor(5)
sr = sklearnRandomForestRegressor()
sklearn_scores = cross_val_scores(sr, X_mod, y, 10)
print "sklearn mean squared error:", sklearn_scores
scores = cross_val_scores(r, X, y, 10)
print "user's mean squared errors:", scores
