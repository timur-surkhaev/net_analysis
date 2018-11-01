# (Python 3.6.5)

import copy
import random 
from itertools import chain

import scipy.stats as ss
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt 

def Random_rewiring_brute_force(G, L, r=5): 
    # Код функции предоставил Поспелов Никита Андреевич
    # (физический факультет МГУ им. Ломоносова)
    
    # параметры: сеть, лист связей, отношение числа пересоединенных пар к числу связей
    global List_of_edges
    Number_of_edges = len(L)
    Number_of_rewired_edge_pairs = Number_of_edges*r
    i=0
    #previous_text = ''

    #print len(set(L).intersection(List_of_edges))
    while i < Number_of_rewired_edge_pairs:
        Edge_index_1 = random.randint(0, Number_of_edges-1)
        Edge_index_2 = random.randint(0, Number_of_edges-1)
        Edge_1 = L[Edge_index_1]
        Edge_2 = L[Edge_index_2]
        [Node_A, Node_B] = Edge_1
        [Node_C, Node_D] = Edge_2
        while (Node_A == Node_C) or (Node_A == Node_D) or (Node_B == Node_C) or (Node_B == Node_D):
            Edge_index_1 = random.randint(0, Number_of_edges-1)
            Edge_index_2 = random.randint(0, Number_of_edges-1)
            Edge_1 = L[Edge_index_1]
            Edge_2 = L[Edge_index_2]
            [Node_A, Node_B] = Edge_1
            [Node_C, Node_D] = Edge_2

        if G.has_edge(Node_A, Node_D) == 0 and G.has_edge(Node_C, Node_B) == 0:
            try:
                try:
                    w_ab = G.get_edge_data(Node_A, Node_B)['weight']
                except:
                    pass
                G.remove_edge(Node_A, Node_B)
            except nx.NetworkXError:
                print ('############################################')
                print (G.has_edge(Node_A, Node_B))
                print ((Node_A, Node_B), 'connection not in graph' )
                print (L.index((Node_A, Node_B)))
                #print List_of_edges.index((Node_A, Node_B))

            try:
                try:
                    w_cd = G.get_edge_data(Node_C, Node_D)['weight']
                except:
                    pass
                G.remove_edge(Node_C, Node_D)
            except nx.NetworkXError:
                print ('############################################')
                print (G.has_edge(Node_C, Node_D))
                print ((Node_C, Node_D), 'connection not in graph' )
                print (L.index((Node_A, Node_B)))
                #print List_of_edges.index((Node_A, Node_B))

            try:
                G.add_edge(Node_A, Node_D, weight = w_ab)
            except:
                G.add_edge(Node_A, Node_D)

            try:
                G.add_edge(Node_C, Node_B, weight = w_cd)
            except:
                G.add_edge(Node_C, Node_B)

            L[Edge_index_1] = (Node_A, Node_D)
            L[Edge_index_2] = (Node_C, Node_B)
            i += 1
    #if (i != 0) and (i % (Number_of_edges//100)) == 0:
    #    text = str(round(100.0*i/Number_of_rewired_edge_pairs, 0)) + "%"
    #    if text != previous_text:
    #        print (text)
            #pass
    #    previous_text = text

    G_rewired = nx.empty_graph()
    G_rewired = copy.deepcopy(G)
    return G_rewired
    del L; del G
    del Edge_1; del Edge_2
    del Node_A; del Node_B; del Node_C; del Node_D

def make_triplets_list(df):
    triplets = []
    for i in range(len(df.index)):
        triplet = []
        for j in range(4):
            triplet.append(df.iloc[i,j])
        triplets.append(triplet) 
    #print(triplets)
    return(triplets)

def make_paths(graph, triplet_list):
    all_paths = []
    for triplet in triplet_list:
        #print(triplet)
        triplet_paths = []
        for i in range(3):
            triplet_paths.append(len(nx.shortest_path(graph, triplet[i], triplet[3]))-1)
        #print(triplet_paths)
        all_paths.append(triplet_paths)
    return(all_paths)
   
def describe_graph(graph):
    print('Is directed: ', nx.is_directed(graph))
    print('Nodes:', graph.number_of_nodes())
    print('Edges:', graph.number_of_edges())
    print('Density:', nx.density(graph))
    print('Transitivity', nx.transitivity(graph))
    
def describe_stat(descr):
    descr = list(ss.describe(descr))
    print('Mean: %.2f \n' % descr[2] +
          'SD: %.2f\n' % descr[3] + 
          'Min: %i\n' % descr[1][0] + 
          'Max: %i\n' % descr[1][1])

def paths_hist(paths, colour, title, bins = range(0,9)):
    plt.hist(list(chain(*paths)), 
         bins = bins, 
         align = 'left',
         alpha = 0.2,
         color = colour)
    plt.title(title,fontsize=16)  
    plt.show()
   
def t_test(set_1, set_2):
    t_t = list(ss.ttest_ind(set_1, set_2))
    print('t-критерий: %.3f' % t_t[0], end = '\t')
    print('p-value: %.5f' % t_t[1])


def triplets_analysis(graph, hard_tripls, easy_tripls):    
    hard_paths = make_paths(graph, hard_tripls)
    easy_paths = make_paths(graph, easy_tripls)
    
    hard_paths_unlisted = list(chain(*hard_paths))
    easy_paths_unlisted = list(chain(*easy_paths))
    
    # Гистограммы путей
    paths_hist(hard_paths, 'red', 'Hard Triplets')
    describe_stat(hard_paths_unlisted)
    paths_hist(easy_paths, 'blue', 'Easy Triplets')
    describe_stat(easy_paths_unlisted)

    print('\n' + '=' * 45)
    print('Сравниваем длины путей сложных и простых слов\n')
    t_test(hard_paths_unlisted, easy_paths_unlisted)
    print('-' * 45)
    
    return(hard_paths_unlisted, easy_paths_unlisted)



# Строим граф ассоциаций
net=pd.read_excel('assoc_eng2.xlsx')
G0=nx.from_pandas_edgelist(net,'source', 'target', 
                           edge_attr=True, 
                           create_using=nx.DiGraph())

# Рандомизируем граф ассоциаций
G0_rewired = copy.deepcopy(G0)
G0_rewired = Random_rewiring_brute_force(G0_rewired, 
                                         list(G0.edges()), 
                                         10)
# Смотрим какие параметры изменились
print('\n========\n' + 'Original' + '\n--------')
describe_graph(G0)
print('\n=======\n' + 'Rewired' + '\n-------')
describe_graph(G0_rewired)

# Читаем данные теста Медника
df=pd.read_excel('eng_rat.xlsx') # for rat test with words in G

# Готовим триплеты для анализа
df_clear = df.dropna()

time_col = '30s_acc'
# заменить на 15 сек.

# делим триплеты по медиане процента справившихся 
median = np.median(df_clear[time_col])
df_hard = df_clear.loc[(df_clear[time_col] < median)].iloc[:,0:4]
df_easy = df_clear.loc[(df_clear[time_col] >= median)].iloc[:,0:4]

# создаём списки триплетов для последующего анализа
hard_triplets = make_triplets_list(df_hard)
easy_triplets = make_triplets_list(df_easy)

# ====================
# Приступаем к анализу
# --------------------

# Гистограммы, описательные статистики и t.test
# triplets_analysis возвращает unlisted списки путей
paths_original = triplets_analysis(G0, hard_triplets, easy_triplets)
print('=' * 40 + '\n' + '=' * 40 + '\n\t    Rewired Graph\n' + 
      '=' * 40 + '\n' + '=' * 40)
paths_rewired = triplets_analysis(G0_rewired, hard_triplets, easy_triplets)

# Сравниваем влияние рандомизации на пути в 
# группах слов тяжёлых и лёгких триплетов

diff_hard = [p1 - p2 for p1, p2 in zip(paths_original[0], paths_rewired[0])]
diff_easy = [p1 - p2 for p1, p2 in zip(paths_original[1], paths_rewired[1])]


print('\n' + '=' * 40)
print('=' * 40)
print('Влияние рандомизации на разность длин \n' + 
      'путей в простых и сложных триплетах: \n')

t_test(diff_hard, diff_easy)
print('-' * 40)


# дополнительно
print('\n' + '=' * 40 + '\n' + '=' * 40)
print('Число компонент сильной связности')
print('Исходный граф: ' + 
      str(len(list(nx.strongly_connected_components(G0)))))
print('Рандомизированный граф: ' + 
      str(len(list(nx.strongly_connected_components(G0_rewired)))))


