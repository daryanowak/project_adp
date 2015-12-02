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
        if len(m) == 0:
            return
        elif self.check_last(m): #sprawdzenie czy wszystkie klasyfikacje sa poprawne
             self.classification = (1, m[0][-1] )
             print self.classification
        elif len(m) == 2: #pozostaja dwa wiec ciezko rozroznic na podstawie indexu Giniego.
            i = 0
            while m[0][i] == m[1][i]:
                i +=1
            self.nr_index = i
            self.value = m[0][i]
            self.left = Node(None, None, None,None )
            self.right = Node(None, None, None , None)
            self.left.insert(m[0]), self.right.insert(m[1])
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
            left_branch = [x for x in m if x[index_G[1]] < index_G[2] ]
            right_branch = [x for x in m if x[index_G[1]] >= index_G[2] ]
            print m, '\n', left_branch, right_branch, '\n'
            self.nr_index = index_G[1]
            self.value = index_G[2]
            self.left = Node(None, None, None, None )
            self.right = Node(None, None, None, None )
            return self.left.insert(left_branch), self.right.insert(right_branch)
        print "koniec"
            

    def check_last(self, m):
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
        j = 0
        mini = (1,0,0)
        while j <= len(matrix[0]) -2:
            tmpl = [ [x[j], x[-1]] for x in matrix]
            tmpl.sort(key = lambda x : x[0]) # sort only by 1st argument of list inside the list
            mini_tmp = self.index_Giniego_wektor_liczby([x[1] for x in tmpl ])
            print mini_tmp , j
            if mini[0] > mini_tmp[0]:
                mini = (mini_tmp[0], j, matrix[mini_tmp[1]][j] )
            j += 1
        return mini
            
    def index_Giniego_wektor_liczby(self, wektor):
        """Dostaje wektor klasyfikacji posortowanych po wartosciach,
            tzn. dostaje ostatnia kolumne w ktorej znajduja sie predykcje
            OUT minimalny index wraz z pozycja"""
        n = len(wektor)
        warianty = [wektor.count( wektor[0])]
        warianty.append(n - warianty[0])
        mini = (1 , None) # (minimalny index Giniego, pozycja na liscie)
        i = 0
        tmpl =[0,0]
        while i < n-1 :
            if wektor [i] == wektor [0]:
                tmpl[0] += 1
            else:
                tmpl[1] += 1
            j = i +1.
            l0 = tmpl[0]/j
            l1 = tmpl[1]/j
            r0 = (warianty[0] - tmpl[0])/(n-j)
            r1 = (warianty[1] - tmpl[1])/(n-j)
            tmp = j/n * ( l0 * ( 1- l0) + l1 * ( 1- l1 ) ) + (n-j)/n * ( r0 * ( 1- r0) + r1 * ( 1-r1) )
            print 'tmp', tmp, 'i', i, 'j', j  , 'n', n ,'l', l0, l1 , 'r', r0, r1 
            if mini[0] > tmp:
                mini = (tmp, i)
            if tmp == 0:
                break
            i += 1
        return mini
    
    def go_through (self, m):
        """Funkcja dostaje wektor z wartosciami przy pomocy ktorych musi zostac klasyfikowany dojendej z dowch grup"""
        if self.probability != 0:
            return self.classification
        elif m[self.nr_index] >= self.value:
            self.left.go_through(m)
        elif m[self.nr_index] < self.value:
            self.right.go_through(m)
        else:
            return "Nie idzie"

class Tree():

    def __init__(self):
        self.root = Node (self, None, None, None)

    def insert (self, matrix):
        print 'halo'
        self.root.insert(matrix)
    
    def go_through(self, m):
        print self.root.go_through(m)

test = [ [1 ,3, 4,True] , [1 ,3, 4,True], [ 1,2,3, False], [1,2,5, True] ]

d =Tree()
d.insert(test)