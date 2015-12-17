import numpy as np
import itertools 
import copy 

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
        ret = "\t"+repr([(row, input_matrix[row][-1]) for row in self.rows])+"\n"
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
        #zwraca 3 elemenetowa liste: [wiesz, kolumna, wartosc wg ktorej dzielimy]
        if not self.criterium(node.rows, node.random_features): #jesli min_gini wychodzi dla podzialu kiedy w jednym lisciu jest zero elementow to
            node.decision = self.major_decision(node.rows) #funkcja criterium zwroci false => podany node jest lisciem wyjdz z funkcji, w innym przypadku criterium 
            #print "node.decision", node.decision
            #print "leaf was created with rows", node.rows
            return 
        #zakladamy ze jesli criterium zwraca krotke to jest mozliwy podzial na dwa nody
        node.indexes = self.criterium(node.rows, node.random_features) # each time new tree has new list_of_permuted_rows  (row, column, value)
        node.left = Node(None, None, [], np.random.choice(range(len(self.permutated_matrix[0])-1), size = n_features, replace = False) )
        node.right = Node(None, None, [], np.random.choice(range(len(self.permutated_matrix[0])-1), size = n_features, replace = False) )
        #przechowywane w root jako indexes
        selected_feature_column = node.indexes[1] #kolumna
        value_in_node = node.indexes[2] #wartosc wg ktorej dzielimy

        for row in node.rows:
            if self.compare(selected_feature_column, self.permutated_matrix[row][selected_feature_column], value_in_node): #if true go left else go right
                node.left.rows.append(row)
            else:
                node.right.rows.append(row)
        self.createNodes(node.left)
        self.createNodes(node.right)

    def criterium(self, rows, columns):
        if called_class == "regression":
            return self.rss(rows, columns)
        elif called_class == "classification":
            return self.gini(rows, columns)

    def gini(self, rows, features):
        node_decisions = [self.permutated_matrix[row][-1] for row in rows]
        if sum(node_decisions) == 0 or sum(node_decisions) == len(node_decisions): #jesli same false albo true
            return False #ten node nie potrzebuje dzielenia jest lisciem
        gini = 1000
        indexes = []
        best_left_decisions = []
        best_right_decisions = []
        n = float(len(rows))
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
            if float(sum(decisions))/len(decisions) > 0.5:
                return True
            else:
                return False 
        elif called_class == "regression":
            return float(sum(decisions)) /len(decisions)

    def go_through(self, node, row):
        if node.left is None:
            return node.decision
        else:
            feature_column = node.indexes[1]
            if input_features_type[feature_column] == "number": #zmienne typu liczbowego
                if row[feature_column] <= node.indexes[2]:
                    return self.go_through(node.left, row)
                else:
                    return self.go_through(node.right, row)
            elif input_features_type[feature_column] == "mixed": #zmienne typu wyliczeniowego
                if row[feature_column] == node.indexes[2]:
                    return self.go_through(node.left, row)
                    #print "ide wlewo"
                else:
                    return self.go_through(node.right, row)
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
        self.input_matrix = None
    
    def fit(self, X, y):
        """uczy klasyfikator na zbiorze treningowym"""
        global called_class #
        called_class = "classification"
        global n_features
        global input_features_type 
        global input_matrix

        M = self.konwerter(X,y)
        input_matrix = M

        input_features_type = self.checkFeaturesType(M) 
        print "input_features_type", input_features_type
        n_features = self.n_features
        self.input_matrix = M
        self.random_forest = self.build_random_forest()

    def konwerter(self, X,y):

        for row,row2 in zip(X,Y):
                row.append(row2)

        return X

    def checkFeaturesType(self, X):
        #matrix wypelniona none o jedna kolumne mniejsza bedzie przechowywac false jesli int or float, 
        #oraz true jesli inaczej -> suma kolumny 0 tylko jesli same liczby do list_of_feauters dodaje "number"
        feature_type_matrix = [[None for i in range(len(X[0])-1)] for j in range(len(X))] 
        for row, vector in  enumerate(X):
            for column, value in enumerate(vector[:-1]):
                if type(X[row][column]) == int or type(X[row][column]) == float:
                    feature_type_matrix[row][column] = False #zmienne typu liczbowego w kolumnie
                else:
                    feature_type_matrix[row][column] = True #zmienne typu wyliczeniowwgo w kolumnie

        list_of_features = []
        for column in range(len(feature_type_matrix[0])):
            summ = 0
            for row in range(len(feature_type_matrix)): #sprawdzenie typu wartosci w kolumnach, suma wartosci True i False w kolumnach
                summ += feature_type_matrix[row][column]
            if summ == 0: #kolumna typu liczbowego (suma False)
                list_of_features.append("number")
            else:
                list_of_features.append("mixed") #kolumna typu wyliczeniowego (suma False i True)
 
        return list_of_features


    def predict(self, X):
        """przewiduje najbardziej prawdopodobne klasy przykladow w X; wraca wektor dlugosci m. 
        Pobiera macierz przykladowych wektorow bez decyzji, przepuszcza przez kazde drzewo self.random_forest 
        i generuje najbardziej prawdopodobna decyzje. Wynikiem jest wektor dlugosci macierzy wejsciowej."""
        final_decision = []
        for v in X: #sprawdzamy decyzje wiekszosciowa, prog 0.5 
            decyzje =[tree.go_through(tree.root, v) for tree in self.random_forest] 
            if decyzje.count(True)/float(len(decyzje)) > 0.5:
                final_decision.append(True)
            else:
                final_decision.append(False)          
        return final_decision

    def predict_proba(self, X):

        """Zwraca prawdopodobienstwo przynaleznosci przykladow z X do pierwszej klasy(major_decision)"""

        all_decisions = [] #tablica wszystkich decyzji dla kazdego wiersza i drzewa

        for row in X:
            decisions = []
            for tree in self.random_forest:
                if go_through(tree.root, row):
                    decison = True
                else:
                    decision = False
                decisions.append(decison)

            all_decisions.append(decisions)

        for decision in all_decisions[0]:
            if float(sum(all_decisions[0]))/len(all_decisions) > 0.5:
                major_decision = True #major_decision w 0 wiersz = pierwsza klasa
            else:
                major_decision = False

        proba_lista = []

        for decisions in all_decisions:
            if major_decision: #if True
                proba_lista.append(float(sum(decisons))/ len(decisons))
            else: #if False
                proba_lista.append(1-(float(sum(decisons))/ len(decisons)))

        return proba_lista

    
    def build_random_forest(self):
        """buduje las losowy. Tworzy 11 pierwszych drzew i sprawdza stabilizacje bledu OOB. W przypadku braku stabilizacji powieksza las"""
        counter = 0
        while counter < 30:
            counter += 1
            self.random_forest.append(self.buildTree())#budowanie drzewa, losowanie wierszy w buildTree 

        while self.find_ooberr() > 0.01:  #ooberr liczymy dla ostatnich 10 drzew lasu
            print "obliczam blad i buduje dodatkowe drzewo"
            counter += 1
            self.random_forest.append(self.buildTree()) 
        print "Random Forest was build, has %d trees" % len(self.random_forest)



    def buildTree(self):
        """Funkcja zwraca drzewo zbudowane na podstawie losowowo zbudowanej macierzy(submatrix). 
        Losowanie wierszy ze zwracaniem, zachowujac wysokosc macierzy.
        Sprawdza zmiane stosunku decyzji zeby zapobiec pobieraniu jednotypowych decyzji"""
        permutated_matrix = []
        rows_random = np.random.choice(range(len(self.input_matrix)), size = len(self.input_matrix), replace = True)
        rows_random.sort() #losuje wiersze nowej tablicy ze zwracaniem 
        out_of_bag = list(set(range(len(self.input_matrix)))-set(rows_random))  #wierszy ktorych nie uzyto do uczenia tego drzewa uzyjemy przy obliczeniu ooberr

        for row in rows_random: 
            permutated_matrix.append(self.input_matrix[row])

        if self.check_decision_proportion(permutated_matrix):  #sprawdza podobienstwo proporcji klas decyzyjnych. 
            tree = Tree(permutated_matrix)                                   #Gdy spelnia zalozenia budowane jest nowe drzewo na podstawie losowo wygenerowanej tablicy.  
            tree.insert(tree.root) #budowanie Node
            tree.out_of_bag = out_of_bag #wierszy ktorych nie uzyto do uczenia tego drzewa uzyjemy przy obliczeniu ooberr
            #print "new tree ####################################################################" + "\n"
            #print tree.root
            #print "drzewo OOB", tree.out_of_bag
            return tree      
        else:
            print "buildTree recursion"
            self.buildTree() #w przypadku, gdy proporcja klas decyzyjnych nie spelnia zalozen powtornie losuje wiersze i kolumny             
        

    def check_decision_proportion(self, permutated_matrix):
        """Sprawdza czy proporcja pomiedzy poszczegolnymi klasami jest podobna do tej w pelnym zbiorze treningowym.  
        Zwraca True jesli proporcja pomiedzy decyzjami jest wieksza rowna 0.5"""
        input_decision_list = []
        output_decision_list = []

        for row in range(len(self.input_matrix)):
            input_decision_list.append(self.input_matrix[row][-1])
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
        

    def find_ooberr(self):
        print "self.random_forest", self.random_forest
        """ kazde drzewo ma self.ooberr { {nr_wierszu : [lista decyzji otrzymnych przy go_through przez 
        drzewa gdzie ten wiersz byl w out_of_bag], ....}, ooberr: wartosc_ooberr_dla_tego_drzewa_uzywana_pozniej_we_wzorze}"""
        for index, tree in enumerate(self.random_forest):
            if tree.ooberr == None: #czyli dla tego drzewa ooberr jeszcze nie byl liczony
                tree.ooberr = self.update(index-1)
        
        oober_10 = self.random_forest[-11].ooberr["ooberr_value"] - sum([self.random_forest[index].ooberr["ooberr_value"] for index in range(-10,0,1)])/10 #sprawdza oober_10 dla 10 ostatnio powstalych drze
        return oober_10


    def update(self, tree_index):
        if tree_index == -1: #musimy po raz pierwszy stworzyc dictionary
            known_ooberr_dict = {"ooberr_dict":{},"ooberr_value":None}
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

        #wybierajac major decision na kazdej liscie i porownujac z decyzja prawidlowa obliczamy ration i zapisujemy do ooberr_value
        list_right_vs_major_decision = []
        for row in known_ooberr_dict["ooberr_dict"]:
            if known_ooberr_dict["ooberr_dict"][row]: #jesli lista decyzji nie jest pusta
                right_decision = self.input_matrix[row][-1]
                #print "known_ooberr_dict[ooberr_dict][%d]" % row, known_ooberr_dict["ooberr_dict"][row]
                trues = sum(known_ooberr_dict["ooberr_dict"][row])
                falses = len(known_ooberr_dict["ooberr_dict"][row]) - trues
                if trues > falses:
                    major_decision = True
                else:
                    major_decision = False
                #print "right_decision", right_decision
                #print "major_decision", major_decision
                list_right_vs_major_decision.append(abs(right_decision - major_decision)) #0 if true (means that right and major decision are the same)

        known_ooberr_dict["ooberr_value"] = float(sum(list_right_vs_major_decision))/len(list_right_vs_major_decision)
        print "known_ooberr_dict", known_ooberr_dict
        return known_ooberr_dict


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

#########################################################################################################################################
#### THE END
############################################################################################################################################
if __name__  == "__main__":
    Test().build_test_string_set()
    RandomForestClassifier(16).fit(X,Y)
