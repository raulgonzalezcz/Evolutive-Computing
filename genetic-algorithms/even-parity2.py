#!/usr/bin/env python
# coding: utf-8
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.random import check_random_state
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# Version 1.13.3 -- pip install numpy==1.13.3
import numpy as np
from IPython.display import Image, display
import pydotplus

# Training samples
X=np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]],dtype=bool)
Y=np.array([1,0,0,1,0,1,1,0], dtype=bool)

x = np.ones(5, dtype=bool)

def logi_and(a,b):
    #return a and b
    return a.astype(bool) & b.astype(bool) 

#def logi_or(a,b):
#    tam=len(a)-1
#    x = np.ones(tam, dtype=bool)
   #return a or b
#    for i in range(tam):
#        x[i]=a[i]|b[i]
    
#    return x
def logi_or(a, b):
    return a.astype(bool) | b.astype(bool)
    
def logi_not(a):
    #return not a
    return ~a.astype(bool) 
        
def logi_xor(a,b):
     return a != b

logic_and = make_function(function=logi_and, name='AND',arity=2)
logic_or = make_function(function=logi_or, name='OR',arity=2)
logic_xor = make_function(function=logi_xor, name='XOR',arity=2)
logic_not = make_function(function=logi_not, name='NOT',arity=1)

#function_set = [logic_and, logic_not,logic_or]
function_set = [logic_and,logic_or,logic_not]
est_gp = SymbolicRegressor(population_size=100,
                           generations=150,
                           #stopping_criteria=0.01,
                           tournament_size=2,
                           function_set= function_set,
                           parsimony_coefficient=0.009,
                           max_samples=1.0,
                           verbose=1,
                           p_crossover=0.9, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.0, p_point_mutation=0.0,
                           n_jobs=-1)


est_gp.fit(X,Y)
print(est_gp._program)
print("-------------------------------")
#print(est_gp._programs)
score_gp = est_gp.score(X, Y)
print(score_gp)
graph = pydotplus.graphviz.graph_from_dot_data(est_gp._program.export_graphviz())
graph.write_svg('test.svg')
#res = Image(graph.create_png())
#display(res)




