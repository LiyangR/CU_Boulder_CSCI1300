import GAinspector
import numpy as np
from utils import *
#from matplotlib import pyplot as plt

def randomGenome(length):
    """
    :param length:
    :return: string, random binary digit
    """
    """Your Code Here"""
    array = np.random.randint(2, size=length)
    return "".join(str(x) for x in array)



def makePopulation(size, length):
    """
    :param size - of population:
    :param length - of genome
    :return: list of length size containing genomes of length length
    """

    """Your Code Here"""

    temp_list = []
    for i in range(size):
        temp_list.append(randomGenome(length))
    return temp_list


def fitness(genome):
    """
    :param genome: 
    :return: the fitness value of a genome
    """
    return sum(int(x) for x in genome)

def evaluateFitness(population):
    """
    :param population: 
    :return: a pair of values: the average fitness of the population as a whole and the fitness of the best individual in the population.
    """
    temp_list = []
    for x in population:
        temp_list.append(fitness(x))
    return Average(temp_list), max(temp_list)
    #return ('%.2f'%Average(temp_list)), ('%.2f'%max(temp_list))



def crossover(genome1, genome2):
    """
    :param genome1:
    :param genome2:
    :return: two new genomes produced by crossing over the given genomes at a random crossover point.
    """
    index = random.randint(1, len(genome1)-1)
    return genome1[:index]+genome2[index:], genome2[:index]+genome1[index:]


def mutate(genome, mutationRate):
    """
    :param genome:
    :param mutationRate:
    :return: a new mutated version of the given genome.
    """
    for i in range(len(genome)):
        if random.random() < mutationRate:
            if genome[i] == "1":
                genome = genome[:i] + "0" + genome[i+1:]
            else:
                genome = genome[:i] + "1" + genome[i+1:]
    return genome

def selectPair(population):
    """

    :param population:
    :return: two genomes from the given population using fitness-proportionate selection. This function should use weightedChoice, which we wrote in class, as a helper function.
    """
    weights = []
    for x in population:
        weights.append(fitness(x))
    return weightedChoice(population, weights), weightedChoice(population, weights)

def runGA(populationSize, crossoverRate, mutationRate, logFile=""):
    """

    :param populationSize: :param crossoverRate: :param mutationRate: :param logFile: :return: xt file in which to
    store the data generated by the GA, for plotting purposes. When the GA terminates, this function should return
    the generation at which the string of all ones was found.is the main GA program, which takes the population size,
    crossover rate (pc), and mutation rate (pm) as parameters. The optional logFile parameter is a string specifying
    the name of a te
    """
    f = open(logFile, 'w')
    
    data_list = makePopulation(populationSize, 20)
    gene_list = []
    avgfit_list = []
    for i in range(50):
        next_g1 = []
        while len(next_g1) < len(data_list):
            g1, g2 = selectPair(data_list)
            if random.random() < crossoverRate:
                c1, c2 = crossover(g1, g2)
                next_g1.append(c1)
                next_g1.append(c2)
            else:
                next_g1.append(g1)
                next_g1.append(g2)
        next_g2 = []
        for j in next_g1:
            next_g2.append(mutate(j, mutationRate))
        avg, best = evaluateFitness(next_g2)
        gene_list.append(i)
        avgfit_list.append(float(avg))
        f.writelines([str(i) + " "+ str(avg) + " " + str(best), "\n"])
        for j in next_g2:
            if fitness(j) == 20:
                return next_g2, i, gene_list, avgfit_list
        data_list = next_g2
    
    return data_list, 49, gene_list, avgfit_list

    
        




if __name__ == '__main__':
    #Testing Code
    print("Test Suite")
    GAinspector.test(randomGenome)
    GAinspector.test(makePopulation)
    GAinspector.test(fitness)
    GAinspector.test(evaluateFitness)
    GAinspector.test(crossover)
    GAinspector.test(mutate)
    GAinspector.test(selectPair)
    
    runGA(100, 0.7, 0.001, "run1.txt")
# =============================================================================
#     max_g = 0
#     min_g = 10000
#     avg = []
#     ran = random.sample(range(0, 50), 5)
#     plt.figure(figsize=(20,20))
#     for i in range(50):
#         li, gene, generation, average = runGA(200, 0.7, 0.001, "run1.txt")
#         if i in ran:
#             #lists = sorted(perf.items())
#             #x, y = zip(*lists) 
#             #print(lists)
#             print(generation)
#             print(average)
#             plt.plot(generation, average)
#         avg.append(gene)
#         if gene > max_g:
#             max_g = gene
#         if gene < min_g:
#             min_g = gene
#     print("Average is: " + str(Average(avg)) + "\n" + "Maximum is: " + str(max_g) + "\n" + "Minimum is: " + str(min_g))
#     
#     
#     plt.xlabel("Generation")
#     plt.ylabel("Average Fitness")
#     plt.title("Average Fitness vs Generation")
#     plt.show()
# =============================================================================