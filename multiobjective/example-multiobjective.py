#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import array
import logging
import random

import numpy
import math

from deap import algorithms
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

import matplotlib.pyplot as plt

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

def viennet(individual, lbda = 0.85):
    """ 
    Implements the test problem Viennet
    Num. variables = 2; bounds in [-1.5, 1.5]; num. objetives = 2.
    @author Raúl González Cruz
    """
    d  = lbda * math.exp(-(individual[0] - individual[1]) ** 2) 
    # Viennet optimization functions
    f1 = 0.5 * ( math.pow(individual[0],2) + math.pow(individual[1],2) ) + math.sin( math.pow(individual[0],2) + math.pow(individual[1],2) )
    f2 = ( math.pow(3*individual[0] - 2*individual[1] + 4, 2) / 8 ) + ( math.pow(individual[0] - individual[1] + 1, 2) / 27 ) + 15
    f3 = (1/ ( math.pow(individual[0],2) + math.pow(individual[1],2) + 1)) - 1.1*math.exp(- ( math.pow(individual[0],2) + math.pow(individual[1],2) ) )
    return f1, f2, f3

# Weights elements per function
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0))
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("attr_float", random.uniform, -3, 3)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 3)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def checkBounds(min, max):
    def decorator(func):
        def wrappper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                for i in range(len(child)):
                    if child[i] > max:
                        child[i] = max
                    elif child[i] < min:
                        child[i] = min
            return offspring
        return wrappper
    return decorator

toolbox.register("evaluate", viennet)
toolbox.register("mate", tools.cxBlend, alpha=1.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=3, indpb=0.3)
toolbox.register("select", tools.selSPEA2)

toolbox.decorate("mate", checkBounds(-3, 3))
toolbox.decorate("mutate", checkBounds(-3, 3)) 

def main():
    random.seed(64)

    MU, LAMBDA = 50, 50
    pop = toolbox.population(n=MU)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)
    
    algorithms.eaMuPlusLambda(pop, toolbox, mu=MU, lambda_=LAMBDA, 
                              cxpb=0.5, mutpb=0.2, ngen=100, 
                              stats=stats, halloffame=hof)

    return pop, stats, hof

if __name__ == "__main__":
    pop, stats, hof = main()
    
    front = numpy.array([ind.fitness.values for ind in pop])
    front = numpy.unique(front, axis=0)
    mitad = int(len(front) / 2)

    # Plot Pareto Front
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    """
    for ind in example_pop:
        plt.plot(ind.fitness.values[0], ind.fitness.values[1], 'k.', ms=3, alpha=0.5)
    """
    for ind in range(len(pop)):
        #plt.plot(ind.fitness.values[0], ind.fitness.values[1], 'bo', alpha=0.74, ms=5)
        ax.scatter3D(pop[ind].fitness.values[0],pop[ind].fitness.values[1],pop[ind].fitness.values[2], label='parametric curve')
    plt.title('Función de Viennet usando dominancia de Fonseca y Fleming')
    ax.set_xlabel('f1(x,y)')
    ax.set_ylabel('f2(x,y)')
    ax.set_zlabel('f3(x,y)')
    plt.show()
