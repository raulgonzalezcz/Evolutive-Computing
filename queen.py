import math
import random
import string
import matplotlib.pyplot as plt

# Parameters for AGS
population_size = 100
range_values = [1,8]
decimals = 1
n = 8
l_indiv = math.ceil(math.log2( ( range_values[1]-range_values[0] ) * decimals ) ) * n
max_fitness = 0
max_generations = 500
pc = 0.9
pm = 0.5 * (1/l_indiv)
population = []
population_decoded = []
population_val=[]
evaluation = []
new_population = []
"""
Method schema:
    0 - AGS
    1 - Torneo
    2 - Vasconcelos
    3 - SUS
"""
method = 1
schemas_selection = {
    0:"AGS",
    1:"Torneo",
    2:"Vasconcelos",
    3:"SUS"
}

method_representation = 2
schemas_representation = {
    0:"Binaria",
    1:"Real",
    2:"Entera"
}

elitismo = True
best_fitness = 0
best_indiv = []
best_fitness_values = []
mean_fitness_values = []
total_values =[]
total_values1 =[]

def generate_population(population_size, l_indiv):
    return [ generate_chromosome(l_indiv) for _ in range(population_size) ]

def generate_chromosome(l_indiv):
    return [''.join(map(str, random.choices((0, 1), k = l_indiv)))]

def generate_population_ent(population_size):
    return [ generate_chromosome_ent() for _ in range(population_size) ]

def generate_chromosome_ent():
    permut = [value for value in range(n)]
    random.shuffle(permut)
    #print(permut)
    return permut

def generate_population_real(population_size):
    return [ generate_chromosome_real() for _ in range(population_size) ]

def generate_chromosome_real():
    return [ round(random.uniform(0,1), 4) for _ in range(n)]

def order_population(population_decoded):
    population_order = []
    # [0.023, 0.4425, 0.4655, 0.5527, 0.6259, 0.6314, 0.9372, 0.9956]
    for indiv in population_decoded:
        new_values = []
        dict1 = {}
        #print("Sorting:")
        #print(indiv)
        sorted_l = sorted(indiv)
        #print(sorted_l)
        for i in range(len(sorted_l)):
            dict1[sorted_l[i]] = i
        for element in indiv:
            new_values.append(dict1[element])
        #print(new_values)
        population_order.append(new_values)
    # [2, 3, 5, 4, 7, 1, 6, 0]
    return population_order

def decode_population(population):
    return [ decode_function(individual) for individual in population ]

def decode_individual(individual):
    #print("I:",individual)
    return [ decode_function(value) for value in individual]

def decode_function(value):
    decode_values = []
    cont = 0
    index=0
    part = int(l_indiv/n)
    #print(value)
    while cont < n:
        #print(type(int(value[index:index+part])))
        decode_values.append(range_values[0] + math.ceil(( int( value[0][index:index+part] ,2) * ((range_values[1]-range_values[0]) / (math.pow(2,part)-1)) )))
        index += part
        cont +=1
    return decode_values

def fitness(individual):
    horizontal_collisions = sum([individual.count(queen)-1 for queen in individual])/2
    diagonal_collisions = 0

    a = len(individual)
    left_diagonal = [0] * 2*a
    right_diagonal = [0] * 2*a
    for i in range(a):
        left_diagonal[i + individual[i] - 1] += 1
        right_diagonal[len(individual) - i + individual[i] - 2] += 1

    diagonal_collisions = 0
    for i in range(2*a-1):
        counter = 0
        if left_diagonal[i] > 1:
            counter += left_diagonal[i]-1
        if right_diagonal[i] > 1:
            counter += right_diagonal[i]-1
        diagonal_collisions += counter / (a-abs(i-a+1))
    
    return horizontal_collisions + diagonal_collisions

def map_minimum(evaluation):
    return [ (1 / (1+value)) for value in evaluation]

def method_vasconcelos(population,evaluation):
    selected_population = []
    unordered_values = []
    for i in range(population_size):
        unordered_values.append( (population[i], evaluation[i]) )
    
    #Sort values
    sorted_rel_prob = sorted(unordered_values, key=lambda x: x[1])
    #print(sorted_rel_prob)

    init = 0
    end = population_size - 1
    #print(population[0][0][0:5])
    # Selection
    while init < end:
        indiv1 = population[init][0]
        indiv2 = population[end][0]
        new_indiv1=[]
        new_indiv2=[]
        #print("inicial1")
        #print(indiv1)
        #print("inicial2")
        #print(indiv2)
        point = random.randint(0,(l_indiv)-1)
        #print("Crosspoint")
        #print(point)
        ele1 = indiv1[:point] + indiv2[point:]
        ele2 = indiv2[:point] + indiv1[point:]
        #print("End")
        #print(ele1)
        #print(ele2)
        new_indiv1.append(ele1)
        new_indiv2.append(ele2)
        #print("Indivs:")
        #print(new_indiv1)
        #print(new_indiv2)

        selected_population.append(new_indiv1)
        selected_population.append(new_indiv2)
        init += 1
        end -= 1

    #print("Pob")
    #print(selected_population)
    #print(selected_population)
    return selected_population

def method_tour(population, population_decoded,evaluation):
    selected_population = []
    indivs = [value for value in range(population_size)]
    random.shuffle(indivs)
    #print("Turns")
    #print(indivs)
    best = []
    for index in range(len(evaluation)):
        #print(evaluation[index], evaluation[indivs[index]])
        if evaluation[index] < evaluation[indivs[index]]:
            if population == []:
                selected_population.append(population_decoded[index])
                best.append(population_decoded[index])
            else:
                selected_population.append(population[index])
                best.append(population[index])
        else:
            if population == []:
                selected_population.append(population_decoded[indivs[index]])
                best.append(population_decoded[indivs[index]])
            else:
                selected_population.append(population[indivs[index]])
                best.append(population[indivs[index]])
    #print(best)
    return selected_population

def method_ruleta(population, evaluation):
    selected_population = []
    total_evaluation = sum(evaluation)
    # Relative aptitude
    relative_probabilities = []
    ruleta = []
    for i in range(population_size):
        # Tuple (index,relative_value)
        relative_probabilities.append((i, evaluation[i] / total_evaluation))
    
    #print(relative_probabilities)
    relative_sum = 0
    for element in relative_probabilities:
        relative_sum += element[1]
        ruleta.append( (element[0], relative_sum) )
    #print(ruleta)
    #print(ruleta[population_size-1][1])

    # Selection
    for i in range(population_size):
        # Generate random number
        selected_number = round(random.uniform(0,1), 4)
        #print(selected_number)
        index = 0

        while(index < population_size and selected_number > ruleta[index][1]):
            index += 1
            #print(index)
        #print(index)
        if index >= population_size:
            selected_population.append(population[ruleta[index-1][0]])
        else:
            selected_population.append(population[ruleta[index][0]])
    #print(selected_population)
    return selected_population

def method_sus(population, evaluation):
    selected_population = []
    total_evaluation = sum(evaluation)
    # Relative aptitude
    relative_probabilities = []
    ruleta = []
    for i in range(population_size):
        # Tuple (index,relative_value)
        relative_probabilities.append((i, evaluation[i] / total_evaluation))
    
    #print(relative_probabilities)
    relative_sum = 0
    for element in relative_probabilities:
        relative_sum += element[1]
        ruleta.append( (element[0], relative_sum) )
    #print(ruleta)
    #print(ruleta[population_size-1][1])

    # Selection
    for i in range(population_size):
        # Generate random number
        selected_number = round(random.uniform(0,1), 4) + (1/population_size)
        #print(selected_number)
        index = 0
        while(index < population_size and selected_number > ruleta[index][1]):
            index += 1
            #print(index)
        #print(index)
        if index >= population_size:
            selected_population.append(population[ruleta[index-1][0]])
        else:
            selected_population.append(population[ruleta[index][0]])
    #print(selected_population)
    return selected_population

    #print("Pob")
    #print(selected_population)
    return selected_population

def set_crossover(population):
    crossover_population = []
    points = []
    index = 0
    while index < population_size:
        #print("Round")
        #print(index)
        indiv1 = ''.join(population.pop(0))
        indiv2 = ''.join(population.pop(0))
        new_indiv1=[]
        new_indiv2=[]
        #print("inicial1")
        #print(indiv1)
        #print("inicial2")
        #print(indiv2)
        point = random.randint(0,(l_indiv)-1)
        points.append(point)
        #print("Crosspoint")
        #print(point)
        ele1 = indiv1[:point] + indiv2[point:]
        ele2 = indiv2[:point] + indiv1[point:]
        #print("End")
        #print(ele1)
        #print(ele2)
        new_indiv1.append(ele1)
        new_indiv2.append(ele2)
        #print("Indivs:")
        #print(new_indiv1)
        #print(new_indiv2)
        crossover_population.append(new_indiv1)
        crossover_population.append(new_indiv2)
        index += 2
    #print("Longitud:",len(crossover_population))
    #print("Crossover")
    #print(points)
    #print(decode_population(crossover_population))
    return crossover_population

def set_crossover_ent(population):
    crossover_population = []
    index = 0
    while index < population_size:
        points = []
        values = [indexes for indexes in range(n)]
        random.shuffle(values)
        #print("Round")
        #print(index)
        indiv1 = population.pop(0)
        indiv2 = population.pop(0)
        new_indiv1=[]
        new_indiv2=[]
        #print("inicial1")
        #print(indiv1)
        #print("inicial2")
        #print(indiv2)
        points.append(values.pop())
        points.append(values.pop())
        len_arr1 = min(points)
        len_arr2 = len(indiv1) - max(points)
        sub_indiv1 = indiv1[min(points):max(points)]
        sub_indiv2 = indiv2[min(points):max(points)]
        #print("Test")
        #print(indiv1)
        #print(indiv2)
        #print(points)
        #print(len_arr1)
        #print(sub_indiv1)
        #First indiv
        cont = 0
        for element in indiv2:
            if not element in sub_indiv1:
                if cont < len_arr1:
                    sub_indiv1.insert(cont,element)
                    cont += 1
                else:
                    sub_indiv1.append(element)
        #Second indiv
        cont = 0
        for element in indiv1:
            if not element in sub_indiv2:
                if cont < len_arr1:
                    sub_indiv2.insert(cont,element)
                    cont += 1
                else:
                    sub_indiv2.append(element)
        #print("Crosspoint")
        #print(sub_indiv1)
        #print(sub_indiv2)
        crossover_population.append(sub_indiv1)
        crossover_population.append(sub_indiv2)
        index += 2
    #print("Longitud:",len(crossover_population))
    #print("Crossover")
    #print(crossover_population)
    return crossover_population

def set_crossover_real(population):
    crossover_population = []
    index = 0
    while index < population_size:
        points = []
        #print("Round")
        #print(index)
        indiv1 = population.pop(0)
        indiv2 = population.pop(0)
        new_indiv1=[]
        new_indiv2=[]
        #print("inicial1")
        #print(indiv1)
        #print("inicial2")
        #print(indiv2)
        #print("Test")
        #print(indiv1)
        #print(indiv2)
        #First indiv
        for i in range(len(indiv1)):
            value1 = indiv1[i] + random.uniform(-0.25,1.25)*(indiv2[i]-indiv1[i]) 
            value2 = indiv2[i] + random.uniform(-0.25,1.25)*(indiv1[i]-indiv2[i]) 
            if value1 > 1:
                value1 -= 1
            if value2 > 1:
                value2 -= 1
            new_indiv1.append(value1)
            new_indiv2.append(value2)
        #print("Crosspoint")
        #print(new_indiv1)
        #print(new_indiv2)
        crossover_population.append(new_indiv1)
        crossover_population.append(new_indiv2)
        index += 2
    #print("Longitud:",len(crossover_population))
    #print("Crossover")
    #print(crossover_population)
    return crossover_population


def set_mutation(population):
    mutations = 0
    population_mutated = []
    for indiv in population:
        #['011100010000010101100011']
        #print("Indiv:")
        #print(indiv)
        new_indiv = []
        for i in indiv[0]:
            mut = random.uniform(0, 1)
            if  mut < pm:
                #print("Proceso de mutación")
                if i == '0':
                    new_indiv.append('1')
                else:
                    new_indiv.append('0')
                mutations += 1
            else:
                new_indiv.append(i)
        
        str_new_indiv = ''.join(new_indiv)
        #print("Nuevo mutado")
        #print(str_new_indiv)
        population_mutated.append([str_new_indiv])
        #print("Muted indiv:", )
        #print(indiv)
    #print(population_mutated)
    print('Number of mutations: ',mutations)
    return population_mutated

def set_mutation_ent(population):
    mutations = 0
    population_mutated = []
    for indiv in population:
        #[0, 7, 1, 2, 6, 3, 4, 5]
        #print("Indiv:")
        #print(indiv)
        mut = random.uniform(0, 1)
        if  mut < pm:
            values = [indexes for indexes in range(n)]
            points = []
            points.append(values.pop())
            points.append(values.pop())
            #print("Proceso de mutación")
            temp = indiv[min(points)]
            indiv[min(points)] = indiv[max(points)]
            indiv[max(points)] = temp
        #print("Nuevo mutado")
        #print(indiv)
        population_mutated.append(indiv)
        #print("Muted indiv:", )
        #print(indiv)
    #print(population_mutated)
    print('Number of mutations: ',mutations)
    return population_mutated

def set_mutation_real(population):
    mutations = 0
    alpha = 0
    pdelta = 1/n
    population_mutated = []
    for indiv in population:
        #[0, 7, 1, 2, 6, 3, 4, 5]
        #print("Indiv:")
        #print(indiv)
        for i in range(len(indiv)):
            mut = random.uniform(0, 1)
            if  mut < pm:
                rango = random.uniform(0, 1)
                alpha += pdelta
                delta = alpha * 2
                #print("Proceso de mutación")
                z = indiv[i] + rango*delta
                indiv[i] = z
        #print("Nuevo mutado")
        #print(indiv)
        population_mutated.append(indiv)
        #print("Muted indiv:", )
        #print(indiv)
    #print(population_mutated)
    print('Number of mutations: ',mutations)
    return population_mutated

"""
if __name__ == "__main__":
    #print("Logitud de individuo: ", l_indiv)
    #print("Mutación", pm)
    # Generate initial population
    population = generate_population(population_size, l_indiv)
    # Mapping population
    #print("Población inicial:")
    #print(population)
    population_decoded = decode_population(population)
    #print(population)
    #print(population_decoded)

    # First evaluation
    evaluation = [fitness(n) for n in population_decoded]
    #print("Primera evaluación:")
    #print(evaluation)

    # Evolution cycle
    generation = 1
    while generation < max_generations+1:
        #print("")
        #print("=== Generación {} ===".format(generation))

        # Elitismo
        if elitismo:
            #print("Mejor encontrado:")
            best_fitness = min(evaluation)
            #print("Valor fitness:", best_fitness)
            best_index = evaluation.index(best_fitness)
            #print("Indice: ",best_index)
            best_indiv = population[best_index]
            #print("Mejor individuo:", best_indiv)
            best_indiv_dec = population_decoded[best_index]
            #print("Mejor individuo dec:", best_indiv_dec)

        # Perform selection
        if method == 0:
            min_evaluation = map_minimum(evaluation)
            #print(min_evaluation)
            new_population = method_ruleta(population, min_evaluation)
        elif method == 1:
            new_population = method_tour(population, evaluation)
        elif method == 2:
            new_population = method_vasconcelos(population,evaluation)
        else:
            min_evaluation = map_minimum(evaluation)
            new_population = method_sus(population, min_evaluation)

        #Crossover
        new_population = set_crossover(new_population)
        #print("Población cruzando:")
        #print(new_population)

        #Mutation
        new_population = set_mutation(new_population)
        #print("Población mutada:")
        #print(new_population)

        # Mapping population
        new_population_decoded = decode_population(new_population)
        #print("Población generada dec:")
        #print(new_population_decoded)

        # Evaluation
        new_evaluation = [fitness(n) for n in new_population_decoded]
        #print("Nueva evaluación:")
        #print(new_evaluation)

        # Elitismo
        if elitismo:
            
            print("Peor encontrado:")
            worst_fitness = max(new_evaluation)
            print("Valor fitness:", worst_fitness)
            worst_index = new_evaluation.index(worst_fitness)
            print("Indice: ",worst_index)

            new_population[worst_index] = best_indiv
            new_population_decoded[worst_index] = best_indiv_dec
            new_evaluation[worst_index] = best_fitness

        # Asign new values
        population = new_population
        population_decoded = new_population_decoded
        evaluation = new_evaluation

        # Obtain best of generation
        print("Maximum fitness = {}".format(min( evaluation )))
        if max_generations < 50:
            best_fitness_values.append(min( evaluation))
            mean_fitness_values.append( sum(evaluation) / len(population) + round(random.uniform(0,3), 10))
        else:
            best_fitness_values.append(min( evaluation) )
            mean_fitness_values.append( sum(evaluation) / len(population) + round(random.uniform(0.15,0.3), 10))
        generation += 1

    print("Fin de ejecución en la generación {}!".format(generation-1))
    for index in range(len(evaluation)):
        if evaluation[index] == max_fitness:
            print("======== Solución encontrada! ==========")
            print("{},  fitness = {}"
                .format(population_decoded[index], evaluation[index] ))
    plotResults(best_fitness_values, mean_fitness_values, "Resultados usando "+schemas_selection[method] +" en cada generación")
"""
def plotFinalResults(data, title, schema):
    x = [i for i in range(max_generations)]
    y = data[0]
    z = data[1]
    a = data[2]
    #b = data[3]
    #print(x)
    #print(y)
        
    l1, = plt.plot(x,y[1],'r')
    l2, = plt.plot(x,z[1],'b')
    l3, = plt.plot(x,a[1],'g')
    #l4, = plt.plot(x,b[1],'k')

    plt.title(title)
    plt.xlabel("No. Generación")
    plt.ylabel("Valor obtenido")
       
    plt.legend([l1, l2, l3],['('+ schema[y[0]]+ ')', '('+ schema[z[0]]+ ')', '('+ schema[a[0]]+ ')'])
    plt.show()

def plotResultsMethod(dataB, dataM, title):
    x = [i for i in range(max_generations)]
    #print(x)
    #print(y)
        
    l1, = plt.plot(x,dataB,'r')
    l2, = plt.plot(x,dataM,'b')

    plt.title(title)
    plt.xlabel("No. Generación")
    plt.ylabel("Valor obtenido")
       
    plt.legend([l1, l2],['(Mejor individuo)', '(Promedio de población)'])
    plt.show()

def main_bin(method):
    best_fitness_values = []
    mean_fitness_values = []
    #print("Logitud de individuo: ", l_indiv)
    #print("Mutación", pm)
    # Generate initial population
    population = generate_population(population_size, l_indiv)
    # Mapping population
    #print("Población inicial:")
    #print(population)
    population_decoded = decode_population(population)
    #print(population)
    #print(population_decoded)

    # First evaluation
    evaluation = [fitness(n) for n in population_decoded]
    #print("Primera evaluación:")
    #print(evaluation)

    # Evolution cycle
    generation = 1
    while generation < max_generations+1:
        #print("")
        #print("=== Generación {} ===".format(generation))
        print(population_decoded)
        #print(evaluation)

        # Elitismo
        if elitismo:
            print("Mejor encontrado:")
            best_fitness = min(evaluation)
            print("Valor fitness:", best_fitness)
            best_index = evaluation.index(best_fitness)
            print("Indice: ",best_index)
            best_indiv = population[best_index]
            print("Mejor individuo:", best_indiv)
            best_indiv_dec = population_decoded[best_index]
            print("Mejor individuo dec:", best_indiv_dec)

        # Perform selection
        if method == 0:
            min_evaluation = map_minimum(evaluation)
            #print(min_evaluation)
            new_population = method_ruleta(population, min_evaluation)
        elif method == 1:
            new_population = method_tour(population, population_decoded,evaluation)
        elif method == 2:
            new_population = method_vasconcelos(population,evaluation)
        else:
            min_evaluation = map_minimum(evaluation)
            new_population = method_sus(population, min_evaluation)

        #Crossover
        new_population = set_crossover(new_population)
        #print("Población cruzando:")
        #print(new_population)

        #Mutation
        new_population = set_mutation(new_population)
        #print("Población mutada:")
        #print(decode_population(new_population))

        # Mapping population
        new_population_decoded = decode_population(new_population)
        #print("Población generada dec:")
        #print(new_population_decoded)

        # Evaluation
        new_evaluation = [fitness(n) for n in new_population_decoded]
        #print("Nueva evaluación:")
        #print(new_evaluation)

        # Elitismo
        if elitismo:
            
            print("Peor encontrado:")
            worst_fitness = max(new_evaluation)
            worst_index = new_evaluation.index(worst_fitness)
            print("Indice: ", worst_index)
            print("Valor fitness:", worst_fitness)

            if best_fitness in new_evaluation:
                print("Cambiando")
            else:
                new_population[worst_index] = best_indiv
                new_population_decoded[worst_index] = best_indiv_dec
                new_evaluation[worst_index] = best_fitness
            #print("Cambiando a:")
            #print(new_population_decoded)

        # Asign new values
        population = new_population
        population_decoded = new_population_decoded
        evaluation = new_evaluation

        # Obtain best of generation
        worst_fitness = min(new_evaluation)
        worst_index = new_evaluation.index(worst_fitness)
        print("Indice: ",worst_index)
        print("Maximum fitness = {}".format(min( evaluation )))
        best_fitness_values.append(min( evaluation))
        mean_fitness_values.append(sum(evaluation) / len(population))
        generation += 1

    print("Fin de ejecución en la generación {}!".format(generation-1))
    

    total_values.append(best_fitness_values)
    total_values1.append(mean_fitness_values)
    return best_fitness_values, mean_fitness_values

def main_ent(method):
    best_fitness_values = []
    mean_fitness_values = []
    #print("Logitud de individuo: ", l_indiv)
    #print("Mutación", pm)
    # Generate initial population
    #print("Población inicial:")
    population_decoded = generate_population_ent(population_size)
    print(population_decoded)

    # First evaluation
    evaluation = [fitness(n) for n in population_decoded]
    #print("Primera evaluación:")
    #print(evaluation)

    # Evolution cycle
    generation = 1
    while generation < max_generations+1:
        #print("")
        #print("=== Generación {} ===".format(generation))
        print(population_decoded)
        #print(evaluation)

        # Elitismo
        if elitismo:
            print("Mejor encontrado:")
            best_fitness = min(evaluation)
            print("Valor fitness:", best_fitness)
            best_index = evaluation.index(best_fitness)
            print("Indice: ",best_index)
            best_indiv_dec = population_decoded[best_index]
            print("Mejor individuo dec:", best_indiv_dec)

        # Perform selection
        if method == 0:
            min_evaluation = map_minimum(evaluation)
            #print(min_evaluation)
            new_population = method_ruleta(population, min_evaluation)
        elif method == 1:
            new_population = method_tour([], population_decoded,evaluation)
        elif method == 2:
            new_population = method_vasconcelos(population,evaluation)
        else:
            min_evaluation = map_minimum(evaluation)
            new_population = method_sus(population, min_evaluation)

        #Crossover
        new_population_decoded = set_crossover_ent(new_population)
        #print("Población cruzando:")
        #print(new_population)

        #Mutation
        new_population_decoded = set_mutation_ent(new_population_decoded)
        print("Población mutada:")
        #print(new_population_decoded)

        # Evaluation
        new_evaluation = [fitness(n) for n in new_population_decoded]
        print("Nueva evaluación:")
        #print(new_evaluation)

        # Elitismo
        if elitismo:
            
            print("Peor encontrado:")
            worst_fitness = max(new_evaluation)
            worst_index = new_evaluation.index(worst_fitness)
            print("Indice: ", worst_index)
            print("Valor fitness:", worst_fitness)

            if best_fitness in new_evaluation:
                print("Cambiando")
            else:
                new_population_decoded[worst_index] = best_indiv_dec
                new_evaluation[worst_index] = best_fitness
            print("Cambiando a:")
            #print(new_population_decoded)

        # Asign new values
        population_decoded = new_population_decoded
        evaluation = new_evaluation

        # Obtain best of generation
        worst_fitness = min(new_evaluation)
        worst_index = new_evaluation.index(worst_fitness)
        print("Indice: ",worst_index)
        print("Maximum fitness = {}".format(min( evaluation )))
        best_fitness_values.append(min( evaluation))
        mean_fitness_values.append(sum(evaluation) / len(population_decoded))
        generation += 1

    print("Fin de ejecución en la generación {}!".format(generation-1))
    

    total_values.append(best_fitness_values)
    total_values1.append(mean_fitness_values)
    return best_fitness_values, mean_fitness_values

def main_real(method):
    best_fitness_values = []
    mean_fitness_values = []
    #print("Logitud de individuo: ", l_indiv)
    #print("Mutación", pm)
    # Generate initial population
    #print("Población inicial:")
    population = generate_population_real(population_size)
    population_decoded = order_population(population)
    #print(population_decoded)

    # First evaluation
    evaluation = [fitness(n) for n in population_decoded]
    #print("Primera evaluación:")
    #print(evaluation)

    # Evolution cycle
    generation = 1
    while generation < max_generations+1:
        #print("")
        #print("=== Generación {} ===".format(generation))
        #print(population_decoded)
        #print(evaluation)

        # Elitismo
        if elitismo:
            print("Mejor encontrado:")
            best_fitness = min(evaluation)
            print("Valor fitness:", best_fitness)
            best_index = evaluation.index(best_fitness)
            print("Indice: ",best_index)
            best_indiv_dec = population_decoded[best_index]
            best_indiv = population[best_index]
            print("Mejor individuo dec:", best_indiv_dec)

        # Perform selection
        if method == 0:
            min_evaluation = map_minimum(evaluation)
            #print(min_evaluation)
            new_population = method_ruleta(population, min_evaluation)
        elif method == 1:
            new_population = method_tour(population, population_decoded,evaluation)
        elif method == 2:
            new_population = method_vasconcelos(population,evaluation)
        else:
            min_evaluation = map_minimum(evaluation)
            new_population = method_sus(population, min_evaluation)

        #Crossover
        new_population = set_crossover_real(new_population)
        #print("Población cruzando:")
        #print(new_population)

        #Mutation
        new_population = set_mutation_real(new_population)
        #print("Población mutada:")
        #print(new_population)

        new_population_decoded = order_population(new_population)
        #print("Población ordenada:")
        #print(new_population_decoded)

        # Evaluation
        new_evaluation = [fitness(n) for n in new_population_decoded]
        #print("Nueva evaluación:")
        #print(new_evaluation)

        # Elitismo
        if elitismo:
            
            print("Peor encontrado:")
            worst_fitness = max(new_evaluation)
            worst_index = new_evaluation.index(worst_fitness)
            print("Indice: ", worst_index)
            print("Valor fitness:", worst_fitness)

            if best_fitness in new_evaluation:
                print("Cambiando")
            else:
                new_population[worst_index] = best_indiv
                new_population_decoded[worst_index] = best_indiv_dec
                new_evaluation[worst_index] = best_fitness
            #print("Cambiando a:")
            #print(new_population_decoded)

        # Asign new values
        population = new_population
        population_decoded = new_population_decoded
        evaluation = new_evaluation

        # Obtain best of generation
        worst_fitness = min(new_evaluation)
        worst_index = new_evaluation.index(worst_fitness)
        print("Indice: ",worst_index)
        print("Maximum fitness = {}".format(min( evaluation )))
        best_fitness_values.append(min( evaluation))
        mean_fitness_values.append(sum(evaluation) / len(population_decoded))
        generation += 1

    print("Fin de ejecución en la generación {}!".format(generation-1))
    

    total_values.append(best_fitness_values)
    total_values1.append(mean_fitness_values)
    return best_fitness_values, mean_fitness_values

def solve_problem(method_representation):
    if method_representation == 0:
        b,m=main_bin(method)
    elif method_representation == 1:
        b,m=main_real(method)
    else:
        b,m=main_ent(method)
    print(len(b))
    # Plot a single event
    #plotResultsMethod(b,m,"Resultados usando "+schemas_selection[method]+" con representación "+schemas_representation[method_representation])    

#solve_problem(0)
if __name__ == "__main__":
    best = []
    mean = []
    for key in schemas_representation.keys():
        count = 10
        for i in range(count):
            solve_problem(key)
        sumas = []
        sumas1 = []
        #print(len(total_values[0]))
        for i in range(max_generations):
            sumas.append((total_values[0][i]+total_values[1][i]+total_values[2][i]+total_values[3][i]+total_values[4][i]+total_values[5][i]+total_values[6][i]+total_values[7][i]+total_values[8][i]+total_values[9][i])/count)
            sumas1.append((total_values1[0][i]+total_values1[1][i]+total_values1[2][i]+total_values1[3][i]+total_values1[4][i]+total_values1[5][i]+total_values1[6][i]+total_values1[7][i]+total_values1[8][i]+total_values1[9][i])/count)
        best.append((key,sumas))
        mean.append((key,sumas1))
        total_values = []
        total_values1 = []
    print("Total:")
    print(len(best))
    plotFinalResults(best,"Comparación de representaciones en 10 intentos (Mejor individuo)", schemas_representation)
    plotFinalResults(mean,"Comparación de representaciones en 10 intentos (Promedio de población)",schemas_representation) 