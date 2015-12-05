import numpy as np

class Node():
    
    def __init__(self, left, right, nr_index, value):
        self.left = left
        self.right = right
        self.nr_index = nr_index
        self.value = value
        self.classification = None

    def insert(self, m):
        """Dodawanie calych macierzy z wartosciami do naszego drzewa,
        macierz ktora wprowadzamy musi miec na pozycji [n][-1] poprana predykcje, gdzie n jest liczba od 0 do len(m).
        Sposob podzialu na podstawie indeksu Giniego"""
        if len(m) == 0 :
            return
        if type(m[0]) != list:
            m= [m]
        if len(m)== 1 or self.check_last(m): #sprawdzenie czy wszystkie klasyfikacje sa
            self.classification = (1, m[0][-1] )
        elif len(m) == 2: #pozostaja dwa wiec ciezko rozroznic na podstawie indexu Giniego.
            i = 0
            while m[0][i] == m[1][i]:
                i +=1
            self.nr_index = i
            self.value = m[0][i]
            self.left = Node(None, None, None,None )
            self.right = Node(None, None, None , None)
            return self.left.insert(m[0]), self.right.insert(m[1])
        elif self.check_identity(m): #wszystkie cechy takie same ale rozna klasyfikacja
            counting = [x[-1] for x in m].count(m[0][-1]) / float(len(m))
            if counting > 0.5:
                 self.classification = ( counting, m[0][-1] )
            else:
                for x in m:
                    if x[-1] != m[0][-1]:
                        self.classification = (counting, x[-1] )
                        break 
        else:
            index_G = self.index_Giniego(m)
            self.value = index_G[2]
            self.nr_index = index_G[1]
            if types(index_G[1]):
                left_branch = [x for x in m if x[index_G[1]] < index_G[2] ]
                right_branch = [x for x in m if x[index_G[1]] >= index_G[2] ]
            else:
                left_branch = [x for x in m if x[index_G[1]] == index_G[2] ]
                right_branch = [x for x in m if x[index_G[1]] != index_G[2] ]
            #print m, '\n', left_branch, right_branch, '\n'
            self.nr_index = index_G[1]
            self.value = index_G[2]
            self.left = Node(None, None, None, None )
            self.right = Node(None, None, None, None )
            return self.left.insert(left_branch), self.right.insert(right_branch)
        print "koniec"
            

    def check_last(self, m):
        #print 'm', m
        tmp = m[0][-1]
        for x in m:
            if x[-1] != tmp:
                return False
        return True

    def check_identity (self, m):
        i = 1
        j= 0
        while j < len (m[0])-1:
            while i < len(m):
                if m[i][j] != m[0][j]:
                    return False            
                i += 1
            j +=1 
        return True

    def index_Giniego(self, matrix): #OUT ( minimalny index Giniego, nr. kolumny dla ktorej parametr jest minimalny, wartosc graniczna)
        """Funkcja ktora ma liczyc index Giniego dla wybranej macierzy, podanej w postaci listy list.
            OUT tuple w postaci ( minimalny index Giniego, nr. kolumny dla ktorej parametr jest minimalny, wartosc graniczna)"""
        global types
        j = 0
        mini = (1,0,0)
        while j <= len(matrix[0]) -2:
            if types[j]:
                tmpl = [ [x[j], x[-1]] for x in matrix]
                mini_tmp = self.index_Giniego_wektor_liczby(tmpl)
            else:
                tmpl = [ [x[j], x[-1]] for x in matrix]
                mini_tmp = self.index_Giniego_str(tmpl)
            if mini[0] > mini_tmp[0]:
                mini = (mini_tmp[0], j, mini_tmp[1])
            j += 1
        return mini
            
    def index_Giniego_wektor_liczby(self, wektorI):
        """Dostaje wektor klasyfikacji posortowanych po wartosciach,
            tzn. dostaje ostatnia kolumne w ktorej znajduja sie predykcje
            OUT minimalny index wraz z pozycja"""
        wektor.sort(key = lambda x : x[0])
        classifications = [x[1] for x in wektor ]
        n = len(classifications)
        warianty = [classifications.count( classifications[0])]
        warianty.append(n - warianty[0])
        mini = (1 , None) # (minimalny index Giniego, pozycja na liscie)
        i = 0
        tmpl =[0,0]
        while i < n-1 :
            if classifications [i] == classifications [0]:
                tmpl[0] += 1
            else:
                tmpl[1] += 1
            j = i +1.
            l0 = tmpl[0]/j
            l1 = tmpl[1]/j
            r0 = (warianty[0] - tmpl[0])/(n-j)
            r1 = (warianty[1] - tmpl[1])/(n-j)
            tmp = j/n * ( l0 * ( 1- l0) + l1 * ( 1- l1 ) ) + (n-j)/n * ( r0 * ( 1- r0) + r1 * ( 1-r1) )
            if mini[0] > tmp:
                mini = (tmp, wektor[i])
            if tmp == 0:
                break
            i += 1
        return mini

    def creat_dict (self, wektor):
        warianty = {}
        for x in wektor:
            if x[0] not in warianty:
                warianty[x[0]] = [0 ,0]
            if x[1] == wektor[0][1]:
                warianty[x[0]][0] += 1
            else:
                warianty[x[0]][1] +=1
        return warianty

    def index_Giniego_str (self, wektor):
        warianty = self.creat_dict(wektor)
        Count0 = [x[1] for x in wektor].count(wektor[0] )
        Count1 = len(wektor) - Count[0]
        mini = (1, None)
        tmpl =[0,0]
        n= len(wektor)
        for x in warianty:
            j = b[x][0] + b[x][1]
            l0 = b[x][0]/j
            l1 = b[x][1]/j
            r0 = (Count0 - b[x][0])/(n-j)
            r1 = (Count1 - b[x][1])/(n-j)
            tmp = j/n * ( l0 * ( 1- l0) + l1 * ( 1- l1 ) ) + (n-j)/n * ( r0 * ( 1- r0) + r1 * ( 1-r1) )
            if mini[0] > tmp:
                mini = (tmp, x)
            if tmp == 0:
                break
        return mini
        
    
    def go_through (self, m):
        """Funkcja dostaje wektor z wartosciami przy pomocy ktorych musi zostac klasyfikowany dojendej z dowch grup"""
        if self.classification != None:
            return self.classification
        if type(self.nr_index):
            if m[self.nr_index] >= self.value:
                return self.left.go_through(m)
            elif m[self.nr_index] < self.value:
                return self.right.go_through(m)
        elif not type(self.nr_index):
            if m[self.nr_index] == self.value:
                return self.left.go_through(m)
            elif m[self.nr_index] != self.value:
                return self.right.go_through(m)
        else:
            return "Nie idzie"

class Tree():

    def __init__(self):
        self.root = Node (self, None, None, None)
        self.oober = None
        self.out_of_bag = None

    def insert (self, matrix):
        global types # (True if int or float; False- str or bool)
        types = []
        for lines in matrix:
            for x in lines:
                value = True
                if type(x) == str or type(x) == bool:
                    value = False
                    break
            types.append(value)
        self.root.insert(matrix)
    
    def go_through(self, m):
        print self.root.go_through(m)


class RandomForestClassifier():

    def __init__(self, n_features):
        self.n_features = n_features
        self.random_forest = []
    
    def fit(self, X):

        """uczy klasyfikator na zbiorze treningowym"""

        self.random_forest = self.build_random_forest(X)

    def predict(self, X):

        """przewiduje najbardziej prawdopodobne klasy przykladow w X; wraca wektor dlugosci m. 
    Pobiera macierz przykladowych wektorow bez decyzji, przepuszcza przez kazde drzewo self.random_forest 
    i generuje najbardziej prawdopodobna decyzje. Wynikiem jest wektor dlugosci macierzy wejsciowej."""

        
        final_decision = []
        for v in X: #sprawdzamy decyzje wiekszosciowa, prog 0.5 
            decyzje =[tree.go_through(v) for tree in self.random_forest] 
            if decyzje.count(True)/len(decyzje) > 0.5:
                final_decision.append(True)
            else:
                final_decision.append(False)
        print "final_decision for input matrix is", final_decision          
        return final_decision


    
    def build_random_forest(self, M):

        counter = 0
        while counter < 11:
            counter += 1
            self.random_forest.append(self.random_submatrix(M)) 
        while self.ooberr(self.random_forest, M) > 0.01: # DODAC W DRZEWIE!!!!!
            self.random_forest.append(self.random_submatrix(M)) 
        print "Random Forest was build, has %d trees" % len(self.random_forest)



    def random_submatrix(self, M):

        """Funkcja zwraca drzewo zbudowane na podstawie losowowo zbudowanej macierzy(submatrix). Losowanie wierszy ze zwracaniem, losowanie kolumn bez zwracania zachowujac wysokosc macierzy"""

        new_M = [[None for i in range(self.n_features + 1)] for j in range(len(M))]
        m_random = np.random.choice(range(len(M)), size = len(M), replace = True)
        out_of_bag = list(set(range(len(M)))-set(m_random)) 
        n_random = np.random.choice(range(len(M[0])-1), size = self.n_features, replace = False)

        for row, row_random in enumerate(m_random):
            new_M[row][-1] = M[row_random][-1]
            print "row", row, new_M
            for column,column_random in enumerate(n_random):
                new_M[row][column] = M[row_random][column_random]

        if self.check_decision_proportion(M,new_M): 

            print "new_M with good decision proportion was created",new_M  
            tree = Tree()
            tree.insert(new_M)
            tree.out_of_bag = out_of_bag #ATRYBUT DO KLASY tree
            return tree

        else:
            self.random_submatrix(M)            
        

    def check_decision_proportion(self, M, new_M):

        """Sprawdza czy proporcja pomiedzy poszczegolnymi klasami jest podobna do tej w pelnym zbiorze treningowym.  Zwraca True jesli proporcja pomiedzy decyzjami jest wieksza rowna 0.5"""

        input_decision_list = []
        output_decision_list = []

        for row in range(len(M)):
            input_decision_list.append(M[row][-1])
            output_decision_list.append(new_M[row][-1])

        a = input_decision_list.count(True)/input_decision_list.count(False)
        b = output_decision_list.count(True)/output_decision_list.count(False)

        decision_proportion = abs(a-b)/a

        if decision_proportion >= 0.5:
            print decision_proportion
            return False

        return True
        

    def ooberr(self, random_forest, M):

        for drzewo in self.random_forest:
            if drzewo.ooberr == None:
                tree_dict = {} #DODAC DO ATRYBUTOW DRZEWA
                for row in range(len(M)): 
                    if row in drzewo.out_of_bag:
                        decision = drzewo.go_through(row[:-1]) #wiersz podawany bez decyzji
                        right_decision = row[-1]
                        tree_dict[row] = [1-abs(right_decision-decision), abs(right_decision-decision)]

                sum_ft = 0
                sum_f = 0

                for key in tree_dict:
                    sum_f += tree_dict[key][1]
                    sum_ft = sum_ft + tree_dict[key][1] + tree_dict[key][0]
                
                self.ooberr = sum_f / sum_ft

        oober_10 = self.random_forest[-10].ooberr - sum([drzewo.ooberr for drzewo in random_forest[-10:]])/10

        return oober_10