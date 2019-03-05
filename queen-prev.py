import math
import random
import string
import matplotlib.pyplot as plt

# Parameters for AGS
population_size = 50
range_values = [1,8]
decimals = 1
l_indiv = math.ceil(math.log2( ( range_values[1]-range_values[0] ) * decimals ) )
max_fitness = 28
max_generations = 100
pc = 0.9
pm = 0.01 * (1/l_indiv)
best_value = 0
best_indiv = []
best_fitness_values = []
mean_fitness_values = []

def generate_population(population_size, l_indiv):
    return [ [ generate_chromosome(l_indiv) for _ in range(8)] for _ in range(population_size)]

def generate_chromosome(l_indiv):
    return ''.join(map(str, random.choices((0, 1), k = l_indiv)))

def decode_population(population):
    return [ [ decode_function(value) for value in individual]  for individual in population ]

def decode_function(value):
    return range_values[0] + math.ceil(( int(value,2) * ((range_values[1]-range_values[0]) / (math.pow(2,l_indiv)-1)) ))

def fitness(individual):
    horizontal_collisions = sum([individual.count(queen)-1 for queen in individual])/2
    diagonal_collisions = 0

    n = len(individual)
    left_diagonal = [0] * 2*n
    right_diagonal = [0] * 2*n
    for i in range(n):
        left_diagonal[i + individual[i] - 1] += 1
        right_diagonal[len(individual) - i + individual[i] - 2] += 1

    diagonal_collisions = 0
    for i in range(2*n-1):
        counter = 0
        if left_diagonal[i] > 1:
            counter += left_diagonal[i]-1
        if right_diagonal[i] > 1:
            counter += right_diagonal[i]-1
        diagonal_collisions += counter / (n-abs(i-n+1))
    
    return round(max_fitness - (horizontal_collisions + diagonal_collisions),4)

def method_selection(population):
    selected_population = []
    for index in range(population_size):
        picked = random.randint(0,population_size-1)
        selected_population.append(population[picked])
    return selected_population

def method_vasconcelos(population,evaluation):
    selected_population = []
    total_evaluation = sum(evaluation)
    # Relative aptitude
    relative_probabilities = []
    ruleta = []
    for i in range(population_size):
        # Tuple (index,relative_value)
        relative_probabilities.append((i, evaluation[i] / total_evaluation))
    
    # Generate "ruleta"
    sorted_rel_prob = sorted(relative_probabilities, key=lambda x: x[1])
    print(sorted_rel_prob)
    relative_sum = 0
    for element in sorted_rel_prob:
        relative_sum += element[1]
        ruleta.append( (element[0], relative_sum) )
    print(ruleta[population_size-1][1])

    # Generate random number
    selected_number = round(random.uniform(0,1), 4)
    index = 0

    # Selection
    for i in range(population_size):
        print(selected_number)
        while(index < population_size and selected_number > ruleta[index][1]):
            index += 1
            #print(index)
        selected_population.append(population[ruleta[index][0]])

        # Adjust values
        selected_number += 1/population_size
        if index >=population_size:
            index = 0
        if selected_number > 1:
            selected_number -= 1

    print("Pob")
    print(selected_population)
    return selected_population

def method_tour(population,evaluation):
    selected_population = []
    for index in range(len(evaluation)):
        picked = random.randint(0,population_size-1)
        if evaluation[index] > evaluation[picked]:
            selected_population.append(population[index])
        else:
            selected_population.append(population[picked])
    return selected_population

def method_ruleta(population, evaluation):
    selected_population = []
    total_evaluation = sum(evaluation)
    # Relative aptitude
    relative_probabilities = []
    ruleta = []
    for i in range(len(population)):
        # Tuple (index,relative_value)
        relative_probabilities.append((i, evaluation[i] / total_evaluation))
    
    # Generate "ruleta"
    sorted_rel_prob = sorted(relative_probabilities, key=lambda x: x[1])
    print(sorted_rel_prob)
    relative_sum = 0
    for element in sorted_rel_prob:
        relative_sum += element[1]
        ruleta.append( (element[0], relative_sum) )
    print(ruleta[population_size-1][1])

    # Selection
    for i in range(population_size):
        # Generate random number
        selected_number = round(random.uniform(0,1), 4)
        #print(selected_number)
        index = 0
        while(index < population_size and selected_number > ruleta[index][1]):
            index += 1
            #print(index)
        print(index)
        if index >= population_size:
            selected_population.append(population[ruleta[index-1][0]])
        else:
            selected_population.append(population[ruleta[index][0]])
    print(selected_population)
    return selected_population
    
def set_crossover(population):
    crossover_population = []
    for index in range(0,len(population),2):
        point = random.randint(1,l_indiv-1)
        indiv1 = population[index]
        indiv2 = population[index+1]
        if (point / l_indiv) < pc:
            indiv1 = population[index][:point] + population[index+1][point:]
            indiv2 = population[index+1][:point] + population[index][point:]
        crossover_population.append(indiv1)
        crossover_population.append(indiv2)
    #print(len(crossover_population))
    return crossover_population

def set_crossover_vasconcelos(population):
    crossover_population = []
    for index in range(0,int(len(population)/2)):
        point = random.randint(1,l_indiv-1)
        indiv1 = population[index]
        indiv2 = population[population_size-1-index]
        if (point / l_indiv) < pc:
            indiv1 = population[index][:point] + population[population_size-1-index][point:]
            indiv2 = population[population_size-1-index][:point] + population[index][point:]
        crossover_population.append(indiv1)
        crossover_population.append(indiv2)
    #print(len(crossover_population))
    return crossover_population

def set_mutation(population): 
    mutations = 0
    for indiv in population:
        for element in indiv:
            for i in range(len(element)):
                mut = round(random.uniform(0, 1), 10)
                if  mut < pm:
                    print("Proceso de mutación de {} ", element)
                    if element[i] == '0':
                        element = element[:i] + '1' + element[i+1:]
                    else:
                        element = element[:i] + '0' + element[i+1:]
                    mutations += 1
                    print("a {}", element)
    print('Number of mutations: ',mutations)
    return population

def plotResults(data, title):
    x = [i for i in range(len(data))]
    y = [value for value in data]
    #print(x)
    #print(y)
        
    plt.plot(x,y)
    plt.title(title)
    plt.xlabel("No. Generación")
    plt.ylabel("Valor obtenido")
       
    plt.show()

if __name__ == "__main__":
    print("Logitud de individuo: ", l_indiv)
    # Generate initial population
    population = generate_population(population_size, l_indiv)
    # Mapping population
    population_decoded = decode_population(population)
    print(population)
    print(population_decoded)

    # First evaluation
    evaluation = [fitness(n) for n in population_decoded]
    print(evaluation)

    # Evolution cycle
    generation = 1
    while not 28 in [value_fit for value_fit in evaluation] and generation < max_generations:
        print("=== Generación {} ===".format(generation))

        # Perform selection
        # new_population = method_selection(population)
        # new_population = method_ruleta(population, evaluation)
        # new_population = method_tour(population, evaluation)
        new_population = method_vasconcelos(population,evaluation)

        #Crossover
        # new_population = set_crossover(new_population)
        new_population = set_crossover_vasconcelos(new_population)

        #Mutation
        new_population = set_mutation(new_population)

        # Mapping population
        new_population_decoded = decode_population(new_population)

        # Evaluation
        new_evaluation = [fitness(n) for n in new_population_decoded]

        # Asign new values
        population = new_population
        population_decoded = new_population_decoded
        evaluation = new_evaluation

        # Obtain best of generation
        print("Maximum fitness = {}".format(max( evaluation )))
        best_fitness_values.append(max( evaluation))
        mean_fitness_values.append(sum(evaluation) / len(population))
        generation += 1

    print("Fin de ejecución en la generación {}!".format(generation-1))
    for index in range(len(evaluation)):
        if evaluation[index] == 28:
            println("Solución encontrada!")
            print("{},  fitness = {}"
                .format(population_decoded[index], fitness(population_decoded[index]) ))
    plotResults(best_fitness_values, "Resultados del mejor por cada generación")
    plotResults(mean_fitness_values, "Resultados del promedio por cada generación")           


    print("Indiv:")
    print(individual)
    x = individual[0]
    y = individual[1]
    a = 1. - x
    b = y - x*x
    result = a*a + b*b*100

    return round(result,4)
    