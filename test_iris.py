import drzewo_dodana_regresja
import numpy
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import itertools 
import copy 

#iris = datasets.load_iris()
#X = iris.data[:100,:]
#y = iris.target[:100]
#print X
#print y

class Test():

    def k_mers(self, k=4):
        """Tworzy wszystkie mozliwe kombinacje k-merow"""
        bases = ['A', 'T', 'G', 'C']
        #poszukiwanie tetramerow
        list_of_k_mers = [''.join(p) for p in itertools.product(bases, repeat = k)]
        return list_of_k_mers

    def build_test_string_set(self):
        list_of_4_mers = self.k_mers()
        with open("enhancers_heart.fa", "r") as enhancers:
            enhancers_lines = enhancers.readlines() #lista linijek z pliku
        with open("random.fa", "r") as random:
            random_lines = random.readlines()
        global X
        global Y
        X = [] #tablica zliczen wystapien 4 merow wraz z decyzja na -1 miejscu 
        Y = []
        for sequence in enhancers_lines:
            k_mer_repetition = [sequence.count(a) for a in list_of_4_mers]
            #k_mer_repetition.append(True)
            X.append(k_mer_repetition)
            Y.append(True)
        for sequence in random_lines: 
            k_mer_repetition = [sequence.count(a) for a in list_of_4_mers]
            #k_mer_repetition.append(False)
            X.append(k_mer_repetition)
            Y.append(False)

Test().build_test_string_set()

def get_score(truth, predictions):
    return sum(map(lambda (x, y): 1 if x == y else 0, zip(truth, predictions)))

sklearn_classifier = RandomForestClassifier()
sklearn_classifier = sklearn_classifier.fit(X, Y)
predictions = sklearn_classifier.predict(X)
print "sklearn: %f" % get_score(Y, predictions)

classifier = drzewo_dodana_regresja.RandomForestClassifier(n_features_user=16)
classifier.fit(X,Y)
predictions = classifier.predict(X)
print "custom: %f" % get_score(Y, predictions)

