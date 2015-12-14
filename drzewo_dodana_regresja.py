import numpy as np
import itertools 

class Node():
    def __init__(self, left, right, rows, random_features):
        self.left = left
        self.right = right
        self.rows = rows
        self.random_features = random_features # losowo wybrane na kazdym nodzie kolumny
        self.indexes = None #indeksy wartosci zawartych w macierzy
        self.decision = None #w przypadku regresji bedzie trzymac srednia wartosc a w klasyfikacji true/ false

    def __repr__(self):

        """Printuje utworzone drzewo od korzenia do lisci."""

        ret = "\t"+repr(self.rows)+"\n"
        if self.left != None:
            for child in [self.left, self.right]:
                ret += child.__repr__()
        return ret

############################################################################################################################################################################
#####TREES
############################################################################################################################################################################

class Tree():
    def __init__(self, permutated_matrix):
        #inicjalizacja roota
        self.root = Node(None, None, range(len(permutated_matrix)), np.random.choice(range(len(permutated_matrix[0])-1), size = n_features, replace = False) )
        self.permutated_matrix = permutated_matrix
        self.ooberr = None
        self.out_of_bag = None

    def insert(self, node):
        self.createNodes(node)


    def createNodes(self, node):
        if not self.criterium(node.rows, node.random_features): #jesli min_gini wychodzi dla podzialu kiedy w jednym lisciu jest zero elementow to
            node.decision = self.major_decision(node.rows) #funkcja criterium zwroci false => podany node jest lisciem wyjdz z funkcji 
            #print "leaf was created with rows", node.rows, node.decision, node.left
            return 
        #zakladamy ze jesli criterium zwraca krotke to jest mozliwy podzial na dwa nody
        node.indexes = self.criterium(node.rows, node.random_features) # each time new tree has new list_of_permuted_rows  (row, column, value)
        node.left = Node(None, None, [], np.random.choice(range(len(self.permutated_matrix[0])-1), size = n_features, replace = False) )
        node.right = Node(None, None, [], np.random.choice(range(len(self.permutated_matrix[0])-1), size = n_features, replace = False) )
        #przechowywane w root jako indexes
        selected_feature_column = node.indexes[1] 
        value_in_node = node.indexes[2]

        for row in node.rows:
            if self.compare(selected_feature_column, self.permutated_matrix[row][selected_feature_column], value_in_node): #if true go left else go right
                node.left.rows.append(row)
            else:
                node.right.rows.append(row)
        self.createNodes(node.left)
        self.createNodes(node.right)
        print "node.rows!!!", node.rows

    def criterium(self, rows, columns):
        if called_class == "regression":
            self.rss(rows, columns)
        elif called_class == "classification":
            self.gini(rows, columns)

    def gini(self, rows, features):
        node_decisions = [self.permutated_matrix[row][-1] for row in rows]
        if sum(node_decisions) == 0 or sum(node_decisions) == len(node_decisions): #jesli same false albo true
            return False #ten node nie potrzebuje dzielenia jest lisciem
        gini = 1000
        indexes = []
        best_left_decisions = []
        best_right_decisions = []
        n = len(rows)
        n_Ls = [] #zapisujemy wartosci n_L i n_R zeby zlapac warunek stopu kiedy kazdy mozliwy podzial wedlug cech nie daje rozdzialu 
        n_Rs = []
        #znajdz wartosc gini dla wybranych wierszy i wylosowanych kolumn
        for row in rows:
            for column in features: 
                value_to_compare = self.permutated_matrix[row][column] #wartosc wedlug ktorej dzielimy
                rows_left = []
                rows_right = []
                #porownaj wartosci z innych wierszy w tej kolumnie zeby rozdzielic na lewe i prawe dziecko
                for row2 in rows:
                    if self.compare(column, self.permutated_matrix[row2][column], value_to_compare):
                        rows_left.append(row2)
                    else:
                        rows_right.append(row2)
                n_L = float(len(rows_left))
                n_R = float(len(rows_right)) #float to prevent calkowite dzielenie i zaokroglenie
                left_decisions = [self.permutated_matrix[row][-1] for row in rows_left] #lista true false
                right_decisions = [self.permutated_matrix[row][-1] for row in rows_right]
                n_l0 = sum(left_decisions) #ilosc decyzji True
                n_l1 = n_L - n_l0          #ilosc decyzji False
                n_r0 = sum(right_decisions)
                n_r1 = n_R - n_r0
                if n_L == 0: #nie chcemy dzielenia na jedna strone bo bez sensu wiec sztucznie zwiekszamy wartosc actual_gini
                    #actual_gini = n_R/n*(n_r0/n_R*(1 - n_r0/n_R) + n_r1/n_R*(1 - n_r1/n_R))
                    actual_gini = gini + 100                
                elif n_R == 0:
                    #actual_gini = n_L/n*(n_l0/n_L*(1 - n_l0/n_L) + n_l1/n_L*(1 - n_l1/n_L))
                    actual_gini = gini + 100
                else:
                    actual_gini = n_L/n*(n_l0/n_L*(1 - n_l0/n_L) + n_l1/n_L*(1 - n_l1/n_L)) + n_R/n*(n_r0/n_R*(1 - n_r0/n_R) + n_r1/n_R*(1 - n_r1/n_R))
                if actual_gini < gini:
                    gini = actual_gini
                    indexes = [row, column, value_to_compare]
                    best_left_decisions = left_decisions
                    best_right_decisions = right_decisions

        #print "min gini", gini
        #print "indexes best", indexes
        if [a*b for a,b in zip(n_Ls,n_Rs)] == 0: #jesli mnozenie skalarne daje 0 to znaczy, ze wyczerpala sie mozliwosc podzialu za pomoca wylosowanych cech
            return False
        elif gini == 0: #czyste rozdzielenie na klasy
            return False
        else:
            return indexes #zwraca indexy cech po ktorych nastepuje podzial

    def rss(self, rows, features):
        print "rss called"
        if len(rows) <= 3: #warunek stop nie dzielimy dalej jest to lisc
            return False
        rss = 1000          #nie mozna dawac tu wartosci
        indexes = []
        for row in rows:
            for column in features:
                value_to_compare = self.permutated_matrix[row][column] #wartosc wedlug ktorej dzielimy
                rows_left = []
                rows_right = []
                #porownaj wartosci z innych wierszy w tej kolumnie zeby rozdzielic na lewe i prawe dziecko
                for row2 in rows:
                    if self.compare(column, self.permutated_matrix[row2][column], value_to_compare):
                        rows_left.append(row2)
                    else:
                        rows_right.append(row2) 
                left_decisions = [self.permutated_matrix[row][-1] for row in rows_left]
                yL = sum(left_decisions)/float(len(left_decisions))
                right_decisions = [self.permutated_matrix[row][-1] for row in rows_right]
                yR = sum(right_decisions)/float(len(right_decisions))
                rss_actual = sum([(decision - yL)**2 for decision in left_decisions]) + sum([(decision - yR)**2 for decision in right_decisions]) 
                if rss_actual < rss:
                    rss = rss_actual
                    indexes = [row, column, value_to_compare]

        return indexes

    def compare(self, selected_feature_column, value_to_be_compared, value_to_compare):
        if input_features_type[selected_feature_column] == "number": #zmienne typu liczbowego, rozdzial na zasadzie <=
            if value_to_be_compared <= value_to_compare: #go left
                return True
            else:
                return False
        elif input_features_type[selected_feature_column] == "mixed": #zmienne typu wyliczeniowego, rozdzial na zasadzie ==
            if value_to_be_compared == value_to_compare:
                return True
            else:
                return False
        else:
            return ValueError

    def major_decision(self, rows):
        decisions = [self.permutated_matrix[row][-1] for row in rows] #w przypadku called_class = regresja sa to wartosci, 
        #natomiast w przypadku klasyfikacji sa to false true
        if called_class == "classification":
            if sum(decisions)/len(decisions) > 0.5:
                return True
            else:
                return False 
        elif called_class == "regression":
            return float(sum(decisions)) /len(decisions)

    def go_through(self, node, row):
        if node.left == None:
            return node.decision
        else:
            feature_column = node.indexes[1]
            if input_features_type[feature_column] == "number": #zmienne typu liczbowego
                if row[feature_column] <= node.indexes[2]:
                    self.go_through(node.left, row)
                else:
                    self.go_through(node.right, row)
            elif input_features_type[feature_column] == "mixed": #zmienne typu wyliczeniowego
                if row[feature_column] == node.indexes[2]:
                    self.go_through(node.left, row)
                    #print "ide wlewo"
                else:
                    self.go_through(node.right, row)
                    #print "ide wprawo"
            else:
                print "blaaaaaaaad"

##################################################################################################################################################
####  RANDOM FOREST CLASSIFIER  
##################################################################################################################################################

class RandomForestClassifier():
    def __init__(self, n_features_user):
        self.n_features = n_features_user
        self.random_forest = []
    

    def fit(self, X):
        """uczy klasyfikator na zbiorze treningowym"""
        global called_class 
        called_class = "classification"
        global n_features
        global input_features_type 
        input_features_type = self.checkFeaturesType(X) 
        n_features = self.n_features
        self.random_forest = self.build_random_forest(X)


    def checkFeaturesType(self, X):
        #matrix wypelniona none o jedna kolumne mniejsza bedzie przechowywac false jesli int or float, 
        #oraz true jesli inaczej -> suma kolumny 0 tylko jesli same liczby do list_of_feauters dodaje "number"
        feature_type_matrix = [[None for i in range(len(X[0])-1)] for j in range(len(X))] 
        for row in X:
            for column in row[:-1]:
                if type(row[column]) == int or type(row[column]) == float:
                    feature_type_matrix = False
                else:
                    feature_type_matrix = True

        list_of_features = []
        for column in range(len(X[0]) - 1):
            summ = 0
            for row in range(len(X)):
                summ += X[row][column]
            if summ == 0:
                list_of_features.append("number")
            else:
                list_of_features.append("mixed")
 
        return list_of_features


    def predict(self, X):
        """przewiduje najbardziej prawdopodobne klasy przykladow w X; wraca wektor dlugosci m. 
        Pobiera macierz przykladowych wektorow bez decyzji, przepuszcza przez kazde drzewo self.random_forest 
        i generuje najbardziej prawdopodobna decyzje. Wynikiem jest wektor dlugosci macierzy wejsciowej."""
        final_decision = []
        for v in X: #sprawdzamy decyzje wiekszosciowa, prog 0.5 
            decyzje =[tree.go_through(tree.root, v) for tree in self.random_forest] 
            if decyzje.count(True)/len(decyzje) > 0.5:
                final_decision.append(True)
            else:
                final_decision.append(False)          
        return final_decision


    
    def build_random_forest(self, input_matrix):
        """buduje las losowy. Tworzy 11 pierwszych drzew i sprawdza stabilizacje bledu OOB. W przypadku braku stabilizacji powieksza las"""
        counter = 0
        while counter < 11:
            counter += 1
            self.random_forest.append(self.buildTree(input_matrix)) 
        print self.find_ooberr(input_matrix)
        while self.find_ooberr(input_matrix) > 0.01:  #ooberr liczymy dla ostatnich 10 drzew lasu
            print self.find_ooberr(input_matrix)
            print "obliczam blad i buduje dodatkowe drzewo"
            counter += 1
            self.random_forest.append(self.buildTree(input_matrix)) 
        print "Random Forest was build, has %d trees" % len(self.random_forest)



    def buildTree(self, input_matrix):
        """Funkcja zwraca drzewo zbudowane na podstawie losowowo zbudowanej macierzy(submatrix). 
        Losowanie wierszy ze zwracaniem, zachowujac wysokosc macierzy.
        Sprawdza zmiane stosunku decyzji zeby zapobiec pobieraniu jednotypowych decyzji"""
        permutated_matrix = []
        rows_random = np.random.choice(range(len(input_matrix)), size = len(input_matrix), replace = True)
        rows_random.sort() #losuje wiersze nowej tablicy ze zwracaniem 
        out_of_bag = list(set(range(len(input_matrix)))-set(rows_random))  #wierszy ktorych nie uzyto do uczenia tego drzewa uzyjemy przy obliczeniu ooberr

        for row in rows_random:
            permutated_matrix.append(input_matrix[row])

        if self.check_decision_proportion(input_matrix, permutated_matrix):  #sprawdza podobienstwo proporcji klas decyzyjnych. 
            tree = Tree(permutated_matrix)                                   #Gdy spelnia zalozenia budowane jest nowe drzewo na podstawie losowo wygenerowanej tablicy.  
            tree.insert(tree.root)
            tree.out_of_bag = out_of_bag #wierszy ktorych nie uzyto do uczenia tego drzewa uzyjemy przy obliczeniu ooberr
            #print "new tree ####################################################################" + "\n"
            #print tree.root
            print "drzewo OOB", tree.out_of_bag
            return tree      
        else:
            print "buildTree recursion"
            self.buildTree(input_matrix) #w przypadku, gdy proporcja klas decyzyjnych nie spelnia zalozen powtornie losuje wiersze i kolumny             
        

    def check_decision_proportion(self, input_matrix, permutated_matrix):
        """Sprawdza czy proporcja pomiedzy poszczegolnymi klasami jest podobna do tej w pelnym zbiorze treningowym.  
        Zwraca True jesli proporcja pomiedzy decyzjami jest wieksza rowna 0.5"""
        input_decision_list = []
        output_decision_list = []

        for row in range(len(input_matrix)):
            input_decision_list.append(input_matrix[row][-1])
            output_decision_list.append(permutated_matrix[row][-1])

        if input_decision_list.count(True)==0 or input_decision_list.count(False) == 0:
            print "The training set is unvalid, contain only one decision class"
            return False
        a = float(input_decision_list.count(True))
       
        if output_decision_list.count(True)==0 or output_decision_list.count(False) == 0 :
            print "The output set is unvalid, contain only one decision class"
            return False
        b = float(output_decision_list.count(True))

        decision_proportion = abs(a-b)/a

        if decision_proportion >= 0.1:
            return False

        return True
        

    def find_ooberr(self, M):
        """Sprawdza stabilizacje bledu OOB. Wartosc oober_10 warunkuje zakonczenie procesu uczenia. 
        Tworzy slownik decyzji dla kazdego drzewa,gdzie przechowywana jest liczba poprawnych i blednych decyzji. 
        Kluczem jest index testowanego wiersza, value[0] jest t_i, value[1] f_i. Zwraca oober_10"""

        oober_10 = self.ooberr(-11, M) - sum([self.ooberr(index, M) for index in range(-10,0,1)])/10 #sprawdza oober_10 dla 10 ostatnio powstalych drze
        print "ostatnie 10 wartosci oobeer w lesie", repr([drzewo.ooberr for drzewo in self.random_forest[-10:]])
        return oober_10

    def ooberr(self, tree_index, M):
        drzewo = self.random_forest[tree_index]
        licznik = 0
        for row in drzewo.out_of_bag:
            if drzewo.go_through(drzewo.root, M[row][:-1]): #sprawdzenie decyzji zwracanej przez go_through, True = decyzja w lisciu true
                decision = True #wiersz podawany bez decyzji, po przechodzeniu przez drzewo zwroci decyzje
            else:
                decision = False
            right_decision = M[row][-1]
            licznik += 1-abs(right_decision-decision) #licznik
        mianownik = len(drzewo.out_of_bag)
        if tree_index > 0:
            #drzewo.ooberr teraz bedzie lista dwuelementowa pamietajaca licznik na 0 pozycji i mianownik na 1
            self.random_forest[tree_index].ooberr = [sum(pair) for pair in zip(self.ooberr(tree_index-1, M), [licznik, mianownik])]
        else:
            self.random_forest[tree_index].ooberr = [licznik, mianownik]
        
        ft_ratio = float(self.random_forest[tree_index].ooberr[0])/self.random_forest[tree_index].ooberr[1]
        return ft_ratio
####################################################################################################################################################################################
####RANDOM FOREST REGRESSOR
####################################################################################################################################################################################

class RandomForestRegressor(RandomForestClassifier): #dodane rss do tree
    def __init__(self, n_features_user):
        RandomForestClassifier.__init__(self, n_features_user)


    def fit(self, X):
        """uczy klasyfikator na zbiorze treningowym"""
        global n_features
        global called_class
        called_class = "regression"
        global input_features_type 
        input_features_type = self.checkFeaturesType(X)  
        n_features = self.n_features
        self.random_forest = self.build_random_forest(X)

    def predict(self, X):
        """przewiduje najbardziej prawdopodobne klasy przykladow w X; wraca wektor dlugosci m. 
        Pobiera macierz przykladowych wektorow bez decyzji, przepuszcza przez kazde drzewo self.random_forest 
        i generuje najbardziej prawdopodobna decyzje. Wynikiem jest wektor dlugosci macierzy wejsciowej."""
        final_decision = []
        for v in X: #sprawdzamy decyzje wiekszosciowa, prog 0.5 
            decisions= [tree.go_through(tree.root, v) for tree in self.random_forest]  
            #wiersz decyzji ptrzymany po przechodzeniu po wszystkich drzewach w random forest
            average = sum(decisions)/float(len(decisions))
            final_decision.append(average)
         
        return final_decision


####################################################################################################################################################################################
#### TEST ON ENCHANCERS_HEART and RANDOM
####################################################################################################################################################################################
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

        X = [] #tablica zliczen wystapien 4 merow wraz z decyzja na -1 miejscu 
        #Y = []

        for sequence in enhancers_lines:
            k_mer_repetition = [sequence.count(a) for a in list_of_4_mers]
            k_mer_repetition.append(True)
            X.append(k_mer_repetition)
            #Y.append(True)

        for sequence in random_lines: 
            k_mer_repetition = [sequence.count(a) for a in list_of_4_mers]
            k_mer_repetition.append(False)
            X.append(k_mer_repetition)
            #Y.append(False)

        return X
############################################################################################################################################
#### THE END
############################################################################################################################################
if __name__  == "__main__":
    input_matrix = Test().build_test_string_set()
    RandomForestClassifier(16).fit(input_matrix)
