"""Created by: Darya Karaneuskaya, Agnieszka Sztyler, Adam Zaborowski, Mariia Savenko"""

import copy
import numpy as np
import itertools
import sys


class Node():

    """Stworzenie wierzcholka(node). Kazdy node przechowuje:
        node.left, node.right: odwolanie do lewego i prawego dziecka
        node.rows: numery wierszy, ktore spelniaja warunek podzialu dla danego node
        node.random_features: numery losowo wybranych kolumn(cech) dla ktorych obliczany jest index Giniego/RSS i nastepuje podzial
        node.indexes: [row, column, value_to_compare] indeks i wartosc wg ktorej nastpil podzial w danym node
        node. decision: przyjmuje wartosc None dla node nie bedacych lisciami. Dla klasyfikacji przechowuje decyzje wiekszosciowa w tym lisciu (True/False). 
        Dla regresji przechowuje srednia wartosc w tym lisciu."""

    def __init__(self, left, right, rows, random_features):
        self.left = left
        self.right = right
        self.rows = rows
        self.random_features = random_features
        self.indexes = None
        self.decision = None


############################################################################################################################################################################
#####TREES
############################################################################################################################################################################

class Tree():
    def __init__(self, permutated_matrix):

        """Stworzenie drzewa. Kazde drzewo przechowuje:
        tree.root: odwolanie do korzenia drzewa
        tree.permutated_matrix: permutacja macierzy X otrzymana w wyniku losowania wierszy ze zwracaniem.
        tree.ooberr: wartosc OOB obliczana dla danego drzewa w przypadku klasyfikacji
        tree.out_of_bag: numery wierszy out_of_bag nie wykorzystane podczas uczenia danego drzewa"""

        self.root = Node(None, None, range(len(permutated_matrix)), np.random.choice(range(len(permutated_matrix[0])-1), size=n_features, replace=False))
        self.permutated_matrix = permutated_matrix
        self.ooberr = None
        self.out_of_bag = None


    def insert(self, node):

        """Buduje drzewo rekurencyjnie po przez sprawdzenie warunkow specyficznych dla:
        regresji: ilosc wierszy w node <= 3
        klasyfikacji: brak mozliwosci uzyskania czystego podzialu na klasy lub otrzymanie czystego podzialu.
        Kryterium (self.criterium) podzialu jest wybierane na podstawie wartosci zmiennej 
        globalnej called_class(regression lub classification). Kryterium zwraca False w przypadku,
        gdy podzial nie jest mozliwy, node klasyfikowany jest jako lisc i przechowuje decyzje. 
        W innym przypadku self.criterium zwraca indexes wg ktorego nastapi podzial w tym node."""

        if not self.criterium(node.rows, node.random_features):
            node.decision = self.major_decision(node.rows)
            return
        node.indexes = self.criterium(node.rows, node.random_features)
        random_features_left = np.random.choice(range(len(self.permutated_matrix[0])-1), size=n_features, replace=False)
        random_features_right = np.random.choice(range(len(self.permutated_matrix[0])-1), size=n_features, replace=False)
        node.left = Node(None, None, [], random_features_left)
        node.right = Node(None, None, [], random_features_right)
        selected_feature_column = node.indexes[1]
        value_in_node = node.indexes[2]

        """Wszystkie wartosci z danej kolumny porownujemy z value_in_node 
        i dokonujemy podzialu na lewe (<=) i prawe (>) dziecko."""

        for row in node.rows:
            if self.compare(selected_feature_column, self.permutated_matrix[row][selected_feature_column], value_in_node):
                node.left.rows.append(row)
            else:
                node.right.rows.append(row)

        self.insert(node.left)
        self.insert(node.right)


    def criterium(self, rows, columns):

        if called_class == "regression":
            return self.rss(rows, columns)
        elif called_class == "classification":
            return self.gini(rows, columns)



    def gini(self, rows, features):

        """Oblicza index Giniego wg ktorego wybierana jest najlepsza cecha do podzialu w node.
        Index Giniego nie jest obliczany i funkcja zwraca False (node jest lisciem), 
        gdy otrzymana zostala czysta klasa decyzyjna (tylko True lub tylko False).
        """ 

        node_decisions = [self.permutated_matrix[row][-1] for row in rows]
        if sum(node_decisions) == 0 or sum(node_decisions) == len(node_decisions):
            return False

        """Gini przyjmuje wartosci od 0 do 1. Optymalny podzial odpowiada min wartosci indexu Giniego.
        Porownywane sa wartosci actual_gini z gini. Poczatkowa wartosc gini zostala zawyzona."""

        gini = 10
        indexes = []
        best_left_decisions = []
        best_right_decisions = []
        n = float(len(rows))
        n_Ls = []
        n_Rs = []
        for row in rows:
            for column in features:
                value_to_compare = self.permutated_matrix[row][column] 
                rows_left = []
                rows_right = []

                for row2 in rows:
                    if self.compare(column, self.permutated_matrix[row2][column], value_to_compare):
                        rows_left.append(row2)
                    else:
                        rows_right.append(row2)
                n_L = float(len(rows_left))
                n_R = float(len(rows_right))
                left_decisions = [self.permutated_matrix[row][-1] for row in rows_left]
                right_decisions = [self.permutated_matrix[row][-1] for row in rows_right]
                n_l0 = sum(left_decisions)
                n_l1 = n_L - n_l0
                n_r0 = sum(right_decisions)
                n_r1 = n_R - n_r0

                if n_L == 0: 
                    actual_gini = gini + 100

                elif n_R == 0:
                    actual_gini = gini + 100

                else:
                    actual_gini = n_L/n*(n_l0/n_L*(1 - n_l0/n_L) + n_l1/n_L*(1 - n_l1/n_L)) + n_R/n*(n_r0/n_R*(1 - n_r0/n_R) + n_r1/n_R*(1 - n_r1/n_R))

                if actual_gini < gini:
                    gini = actual_gini
                    indexes = [row, column, value_to_compare]
                    best_left_decisions = left_decisions
                    best_right_decisions = right_decisions

            """Jesli mnozenie skalarne daje 0 to znaczy, ze wyczerpana zostala mozliwosc podzialu za pomoca wylosowanych cech"""

        if [a*b for a, b in zip(n_Ls, n_Rs)] == 0:
            return False
        elif gini == 0:
            return False
        else:
            return indexes



    def rss(self, rows, features):

        """Oblicza blad sredniokwadratowy (RSS) w przypadku regresji.
        Warunkiem zakonczenia podzialow jest liczba wierszy w node < 3."""

        if len(rows) <= 3:
            return False

        rss = 10000000000000
        indexes = []
        for row in rows:
            for column in features:
                value_to_compare = self.permutated_matrix[row][column]
                rows_left = []
                rows_right = []

                for row2 in rows:
                    if self.compare(column, self.permutated_matrix[row2][column], value_to_compare):
                        rows_left.append(row2)
                    else:
                        rows_right.append(row2)
                left_decisions = [self.permutated_matrix[row][-1] for row in rows_left]
                if len(left_decisions) == 0:
                    rss_actual = rss + 100
                else:
                    yL = sum(left_decisions)/float(len(left_decisions))
                right_decisions = [self.permutated_matrix[row][-1] for row in rows_right]
                if len(right_decisions) == 0:
                    rss_actual = rss + 100
                else:
                    yR = sum(right_decisions)/float(len(right_decisions))
                    rss_actual = sum([(decision - yL)**2 for decision in left_decisions]) + sum([(decision - yR)**2 for decision in right_decisions])
                if rss_actual < rss:
                    rss = rss_actual
                    indexes = [row, column, value_to_compare]

        return indexes

    def compare(self, selected_feature_column, value_to_be_compared, value_to_compare):

        """Warunkuje sposob podzialu w node w zaleznosci od typu danych:
        typ number: <=
        typ mixed: =
        Jesli porownywanie wartosci daje wynik:
        True: dany wiersz dodawany jest do lewego syna
        False: dany wiersz dodwany jest do prawego syna"""

        if input_features_type[selected_feature_column] == "number":
            if value_to_be_compared <= value_to_compare:
                return True
            else:
                return False
        elif input_features_type[selected_feature_column] == "mixed":
            if value_to_be_compared == value_to_compare:
                return True
            else:
                return False


    def major_decision(self, rows):

        """Pobiera wiersze wykorzystywane przez dany Node i porownuje decyzje
        dla kazdego wiersza z decyzja w permutated_matrix. Oblicza i zwraca decyzje wiekszosciowa
        w lisciu. Major decision jest obliczane jako suma decyzji True/False, w przypadku regresji
        suma wartosci. Suma decyzji jest iloscia True. Gdy suma dzielona przez dlugosc > 0.5
        to decyzja wiekszosciowa == True."""

        decisions = [self.permutated_matrix[row][-1] for row in rows]
        if called_class == "classification":
            if float(sum(decisions))/len(decisions) > 0.5:
                return True
            else:
                return False
        elif called_class == "regression":
            return float(sum(decisions)) /len(decisions)


    def go_through(self, node, row):

        """Klasyfikacja nowych danych. Przechodzi przez drzewo zaczynajac od korzenia 
        i zwraca decyzje przechowywana w lisciu (node_decisions)."""

        if node.left is None:
            return node.decision
        else:
            feature_column = node.indexes[1]
            if input_features_type[feature_column] == "number":
                if row[feature_column] <= node.indexes[2]:
                    return self.go_through(node.left, row)
                else:
                    return self.go_through(node.right, row)
            elif input_features_type[feature_column] == "mixed":
                if row[feature_column] == node.indexes[2]:
                    return self.go_through(node.left, row)
                else:
                    return self.go_through(node.right, row)

##################################################################################################################################################
####  RANDOM FOREST CLASSIFIER  
##################################################################################################################################################

class RandomForestClassifier():

    """Przyjmuje parametr zdefiniowany przez uzytkownika n_features_user."""

    def __init__(self, n_features_user):
        self.n_features = n_features_user
        self.random_forest = []
        self.input_matrix = None
    
    def fit(self, X, y):

        """Uczy klasyfikator na zbiorze treningowym. Inicjuje budowe lasu."""
        def check_type(lista):
            for element in lista:
                if type(element) != bool:
                    return False
            return True

        if not len(X) == len(y):
            sys.exit("Pierwszy wymiar X i dlugosc y nie sa rowne!")
        if not len(sorted(set(y))) == 2:
            sys.exit("W wektorze decyzji jest wiecej niz dwie klasy!")
        if not check_type(y):
            sys.exit("Decyzje musza byc typu bool")
        global called_class
        called_class = "classification"
        global n_features
        n_features = self.n_features
        global input_features_type
        M = self.konwerter(X, y)
        input_features_type = self.checkFeaturesType(M)
        global input_matrix
        input_matrix = M
        self.input_matrix = M
        self.build_random_forest()

        print "Classifiere - Random Forest Was Build. It has %d trees" % len(self.random_forest)

    def konwerter(self, X, y):

        """Laczy macierz X z decyzjami y, gdzie decyzje dolaczone
        sa do ostatniej kolumnie w macierzy. Reprezentowane sa w przypadku klasyfikacji
        jako 0/1 zamiast False/True."""

        p = np.column_stack([X, y])
        list_of_lists = np.array(p).tolist()
        return list_of_lists

    def checkFeaturesType(self, X):

        """Sprawdzenie wszystich wartosci zawartych w macierzy X pod katem typu danych.
        Generuje macierz zlozona z True/False w zaleznosci od typu danych:
        zmienne typu wyliczeniowego: True
        zmienne typu liczbowego: False.
        Jezeli wartosci w danej kolumnie sa niejednorodne to cala kolumna jest typu mixed.
        Zwraca liste typow danych w poszczegolnych kolumnach."""


        feature_type_matrix = [[None for i in range(len(X[0])-1)] for j in range(len(X))]
        for row, vector in  enumerate(X):
            for column, value in enumerate(vector[:-1]):
                if type(X[row][column]) == int or type(X[row][column]) == float:
                    feature_type_matrix[row][column] = False
                else:
                    feature_type_matrix[row][column] = True

        list_of_features = []
        for column in range(len(feature_type_matrix[0])):
            summ = 0
            for row in range(len(feature_type_matrix)):
                summ += feature_type_matrix[row][column]
            if summ == 0:
                list_of_features.append("number")
            else:
                list_of_features.append("mixed")

        return list_of_features


    def predict(self, X):

        """Przewiduje najbardziej prawdopodobne klasy przykladow w X; wraca wektor dlugosci m.
        Pobiera macierz przykladowych wektorow bez decyzji, przepuszcza przez kazde drzewo self.random_forest
        i generuje najbardziej prawdopodobna decyzje na podstawie decyzji wiekszosciowej.
        Zwraca wektor dlugosci macierzy wejsciowej."""
        if len(X[0]) != len(input_matrix[0])-1:
            sys.exit("Wymiar wiersza nie jest wlasciwy")
        feature_types = self.checkFeaturesType(self.input_matrix)
        for row in range(len(X)):
            for index,element in enumerate(X[row]):
                if feature_types[index] == "mixed":
                    input_column = [self.input_matrix[i][index] for i in range(len(self.input_matrix))]
                    if not element in input_column:
                        sys.exit("Wartosc nie wystepowala w odpowiedniej kolumnie zbioru uczacego")

        X = np.array(X).tolist()
        final_decision = []
        for v in X:
            decyzje = [tree.go_through(tree.root, v) for tree in self.random_forest]
            if decyzje.count(True)/float(len(decyzje)) > 0.5:
                final_decision.append(True)
            else:
                final_decision.append(False)
        return final_decision

    def predict_proba(self, X):

        """Zwraca prawdopodobienstwo przynaleznosci przykladow z X do klasy wystepujacej jako pierwsza."""
        if len(X[0]) != len(input_matrix[0])-1:
            sys.exit("Wymiar wiersza nie jest wlasciwy")
        feature_types = self.checkFeaturesType(self.input_matrix)
        for row in range(len(X)):
            for index,element in enumerate(X[row]):
                if feature_types[index] == "mixed":
                    input_column = [self.input_matrix[i][index] for i in range(len(self.input_matrix))]
                    if not element in input_column:
                        sys.exit("Wartosc nie wystepowala w odpowiedniej kolumnie zbioru uczacego")
        all_decisions = []
        for row in X:
            decisions = []
            for tree in self.random_forest:
                if tree.go_through(tree.root, row):
                    decision = True
                else:
                    decision = False
                decisions.append(decision)
            all_decisions.append(decisions)

        for decision in all_decisions[0]:
            if float(sum(all_decisions[0]))/len(all_decisions) > 0.5:
                major_decision = True
            else:
                major_decision = False

        proba_lista = []
        for decisions in all_decisions:
            if major_decision:
                proba_lista.append(float(sum(decisions))/ len(decisions))
            else: 
                proba_lista.append(1-(float(sum(decisions))/ len(decisions)))
        return proba_lista


    
    def build_random_forest(self):

        """Buduje las losowy. Tworzy 21 pierwszych drzew i w przypadku klasyfikacji sprawdza stabilizacje bledu OOB.
        W przypadku braku stabilizacji powieksza las o kolejne drzewo i ponownie sprawdza stabilizacje OBB.
        Powtarza ten krok do momentu ustabilizowania wartosci OOB."""

        counter = 0
        while counter < 21:
            counter += 1
            self.random_forest.append(self.buildTree())
        if called_class == "classification":
            while self.find_ooberr() > 0.01:
                self.random_forest.append(self.buildTree())

    def buildTree(self):

        """Funkcja zwraca drzewo zbudowane na podstawie losowowo zbudowanej macierzy(submatrix).
        Losowanie wierszy ze zwracaniem, zachowujac wysokosc macierzy.
        Sprawdza zmiane stosunku decyzji True do False aby zapobiec wykorzystywaniu decyzji jednego typu.
        Jezeli stosunek decyzji nie jest zgodny z zalozeniem permutowana macierz generowana jest powtornie.
        Dla kazdego drzewa generowana jest nowa, permutowana macierz przez losowanie wierszy ze zwracaniem.
        Wierze, ktorych nie uzyto do uczenia danego drzewa sa przechowywane w tree.out_of_bag i wykorzystywane do obliczania ooberr.
        Metoda zwraca drzewo."""

        permutated_matrix = []
        rows_random = np.random.choice(range(len(self.input_matrix)), size=len(self.input_matrix), replace=True)
        rows_random.sort()
        out_of_bag = list(set(range(len(self.input_matrix)))-set(rows_random))
        for row in rows_random:
            permutated_matrix.append(self.input_matrix[row])
        if called_class == "classification":
            if self.check_decision_proportion(permutated_matrix):
                tree = Tree(permutated_matrix)                
                tree.insert(tree.root)
                tree.out_of_bag = out_of_bag
                return tree
            else:
                return self.buildTree()
        elif called_class == "regression":
            tree = Tree(permutated_matrix)
            tree.insert(tree.root)
            tree.out_of_bag = out_of_bag
            return tree

    def check_decision_proportion(self, permutated_matrix):

        """Sprawdza czy proporcja pomiedzy poszczegolnymi klasami jest podobna do tej w pelnym zbiorze treningowym.
        Zwraca True jesli proporcja pomiedzy decyzjami jest wieksza rowna 0.5"""

        input_decision_list = []
        output_decision_list = []

        for row in range(len(self.input_matrix)):
            input_decision_list.append(self.input_matrix[row][-1])
            output_decision_list.append(permutated_matrix[row][-1])

        if input_decision_list.count(True) == 0 or input_decision_list.count(False) == 0:
            print "The training set is unvalid, contain only one decision class"
            return False
        a = float(input_decision_list.count(True))
        if output_decision_list.count(True) == 0 or output_decision_list.count(False) == 0:
            print "The output set is unvalid, contain only one decision class"
            return False
        b = float(output_decision_list.count(True))
        decision_proportion = abs(a-b)/a
        if decision_proportion >= 0.1:
            return False

        return True

    def find_ooberr(self):

        """Kazde drzewo ma przypisany slownik tree.ooberr { {nr_wiersza : [lista decyzji otrzymanych w wyniku przejscia przez
        drzewa (go_through) dla wierszy z out_of_bag], ...}, ooberr: wartosc_ooberr_dla_tego_drzewa_uzywana_pozniej_we_wzorze}.
        Zmiana wartosci ooberr wyliczana jest dla ostatnich 20 drzew."""

        for index, tree in enumerate(self.random_forest):
            if tree.ooberr is None: 
                tree.ooberr = self.update(index-1)
        ooberr_20 =  self.random_forest[-21].ooberr["ooberr_value"] - sum([self.random_forest[index].ooberr["ooberr_value"] for index in range(-20, 0, 1)])/20
        return ooberr_20



    def update(self, tree_index):

        """Oblicza ooberr dla kazdego zbudowanego drzewa.
        Tworzy i zwraca slownik known_ooberr_dict tylko w przypadku klasyfikacji.
        """

        if tree_index == -1:
            known_ooberr_dict = {"ooberr_dict":{}, "ooberr_value":None}
        else:
            known_ooberr_dict = copy.deepcopy(self.random_forest[tree_index].ooberr)
        for row in self.random_forest[tree_index+1].out_of_bag:
            tree = self.random_forest[tree_index+1]
            predicted_decision = False

            if tree.go_through(tree.root, self.input_matrix[row][:-1]):
                predicted_decision = True

            if not row in known_ooberr_dict["ooberr_dict"]:
                known_ooberr_dict["ooberr_dict"][row] = []

            known_ooberr_dict["ooberr_dict"][row].append(predicted_decision)

        """Oblicza stosunek prawidlowych decyzji do wszystkich decyzji dla danego wiersza."""

        list_right_vs_major_decision = []
        for row in known_ooberr_dict["ooberr_dict"]:
            if known_ooberr_dict["ooberr_dict"][row]:
                right_decision = self.input_matrix[row][-1]
                trues = sum(known_ooberr_dict["ooberr_dict"][row])
                falses = len(known_ooberr_dict["ooberr_dict"][row]) - trues
                if trues > falses:
                    major_decision = True
                else:
                    major_decision = False

                list_right_vs_major_decision.append(abs(right_decision - major_decision))

        known_ooberr_dict["ooberr_value"] = float(sum(list_right_vs_major_decision))/len(list_right_vs_major_decision)
        return known_ooberr_dict


####################################################################################################################################################################################
####RANDOM FOREST REGRESSOR
####################################################################################################################################################################################

class RandomForestRegressor(RandomForestClassifier): 
    
    """Klasa regresji dziedziczy wiekszosc metod z klasy RandomForestClassifier."""

    def __init__(self, n_features_user):
        RandomForestClassifier.__init__(self, n_features_user)

    def fit(self, X, y):

        def check_type(lista):
            for element in lista:
                if str(element).isdigit():
                    return True
            return False

        """Uczy regresor na zbiorze treningowym. Inicjuje budowe lasu."""
        if not len(X) == len(y):
            sys.exit("Pierwszy wymiar X i dlugosc y nie sa rowne!")
        if not check_type(y):
            sys.exit("Decyzje musza byc numeryczne")
        global n_features
        n_features = self.n_features
        global called_class
        called_class = "regression"

        self.input_matrix = self.konwerter(X,y)

        global input_features_type
        input_features_type = self.checkFeaturesType(self.input_matrix)
        self.build_random_forest()



    def predict(self, X):

        """przewiduje najbardziej prawdopodobne klasy przykladow w X; zwraca wektor dlugosci m.
        Pobiera macierz przykladowych wektorow bez decyzji, przepuszcza przez kazde drzewo self.random_forest
        i oblicza srednia wartosc w lisciu. Wynikiem jest wektor dlugosci macierzy wejsciowej."""
        if len(X[0]) != len(self.input_matrix[0])-1:
            sys.exit("Wymiar wiersza nie jest wlasciwy")
        feature_types = self.checkFeaturesType(self.input_matrix)
        for row in range(len(X)):
            for index,element in enumerate(X[row]):
                if feature_types[index] == "mixed":
                    input_column = [self.input_matrix[i][index] for i in range(len(self.input_matrix))]
                    if not element in input_column:
                        sys.exit("Wartosc nie wystepowala w odpowiedniej kolumnie zbioru uczacego")
        final_decision = []
        for v in X: 
            decisions= [tree.go_through(tree.root, v) for tree in self.random_forest]
            average = sum(decisions)/float(len(decisions))
            final_decision.append(average)

        return final_decision