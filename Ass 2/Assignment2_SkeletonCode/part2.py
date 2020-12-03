import robby
import numpy as np
from utils import *
import random
POSSIBLE_ACTIONS = ["MoveNorth", "MoveSouth", "MoveEast", "MoveWest", "StayPut", "PickUpCan", "MoveRandom"]
rw = robby.World(10, 10)
rwdemo = robby.World(10, 10)
rw.graphicsOff()


def sortByFitness(genomes):
    tuples = [(fitness(g), g) for g in genomes]
    tuples.sort()
    sortedFitnessValues = [f for (f, g) in tuples]
    sortedGenomes = [g for (f, g) in tuples]
    return sortedGenomes, sortedFitnessValues


def randomGenome(length):
    """
    :param length:
    :return: string, random integers between 0 and 6 inclusive
    """

    """Your Code Here"""
    array = np.random.randint(7, size=length)
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

def fitness(genome, steps=200, init=0.50):
    """

    :param genome: to test
    :param steps: number of steps in the cleaning session
    :param init: amount of cans
    :return:
    """
    if type(genome) is not str or len(genome) != 243:
        raise Exception("strategy is not a string of length 243")
    for char in genome:
        if char not in "0123456":
            raise Exception("strategy contains a bad character: '%s'" % char)
    if type(steps) is not int or steps < 1:
        raise Exception("steps must be an integer > 0")
    if type(init) is str:
        # init is a config file
        rw.load(init)
    elif type(init) in [int, float] and 0 <= init <= 1:
        # init is a can density
        rw.goto(0, 0)
        rw.distributeCans(init)
    else:
        raise Exception("invalid initial configuration")

    temp = []
    for j in range(25):
        f = 0
        for st in range(steps):
            index = rw.getPerceptCode()
            i = genome[index]
            if i == "0":
                f += rw.north()
            
            elif i == "1":
                f += rw.south()
            
            elif i == "2":
                f += rw.east()
            
            elif i == "3":
                f += rw.west()
            
            elif i == "4":
                f += rw.stay()
        
            elif i == "5":
                f += rw.grab()
        
            elif i == "6":
                f += rw.random()
        temp.append(f)
        rw.goto(0, 0)
        rw.distributeCans(init)
    
    return Average(temp)
    
    
            

def evaluateFitness(population):
    """
    :param population:
    :return: a pair of values: the average fitness of the population as a whole and the fitness of the best individual
    in the population.
    """
    geno_set, fit_set = sortByFitness(population)
    return round(Average(fit_set), 2), fit_set[-1], geno_set[-1]


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
            if genome[i] == "0":
                genome = genome[:i] + str(random.choice([1, 2, 3, 4, 5, 6])) + genome[i+1:]
            elif genome[i] == "1":
                genome = genome[:i] + str(random.choice([0, 2, 3, 4, 5, 6])) + genome[i+1:]
            elif genome[i] == "2":
                genome = genome[:i] + str(random.choice([1, 0, 3, 4, 5, 6])) + genome[i+1:]
            elif genome[i] == "3":
                genome = genome[:i] + str(random.choice([1, 2, 0, 4, 5, 6])) + genome[i+1:]
            elif genome[i] == "4":
                genome = genome[:i] + str(random.choice([1, 2, 3, 0, 5, 6])) + genome[i+1:]
            elif genome[i] == "5":
                genome = genome[:i] + str(random.choice([1, 2, 3, 4, 0, 6])) + genome[i+1:]
            else:
                genome = genome[:i] + str(random.choice([1, 2, 3, 4, 5, 0])) + genome[i+1:]
    return genome

def selectPair(population):
    """

    :param population:
    :return: two genomes from the given population using fitness-proportionate selection.
    This function should use RankSelection,
    """
    weights = []
    for i in range(len(population)):
        weights.append(i+1)
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
    
    data_list = makePopulation(populationSize, 243)
    maxfit = -10000
    maxgeno = ""
    for i in range(300):
        next_g1 = []
        geno_set, fit_set = sortByFitness(data_list)
        while len(next_g1) < len(data_list):
            if random.random() < crossoverRate:
                g1, g2 = selectPair(geno_set)
                c1, c2 = crossover(g1, g2)
                next_g1.append(c1)
                next_g1.append(c2)
        next_g2 = []
        for j in next_g1:
            next_g2.append(mutate(j, mutationRate))
        avg, bestfit, bestgeno = evaluateFitness(next_g2)
        if maxfit < bestfit:
            maxfit = bestfit
            maxgeno = bestgeno
        if i%10 == 0:
            #rwdemo.demo(maxgeno)
            #rwdemo.graphicsOff(message="")
            print(maxgeno)
            f.writelines([str(i) + " "+ str(avg) + " " + str(bestfit) + " " + str(bestgeno), "\n"])
        data_list = next_g2
    
    return data_list


def test_FitnessFunction():
    f = fitness(rw.strategyM)
    print("Fitness for StrategyM : {0}".format(f))



#test_FitnessFunction()

runGA(100, 1.0, 0.05, "GAoutput.txt")
#runGA(200, 1.0, 0.05, "GAoutput1.txt")
#runGA(100, 0.5, 0.05, "GAoutput2.txt")
#runGA(100, 1.0, 0.02, "GAoutput3.txt")

    
#rw.demo(rw.strategyM)