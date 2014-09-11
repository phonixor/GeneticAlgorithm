
import GA_population
import gene
from datetime import timedelta

class population(GA_population.GA_population):
    """ implementation of GA_population

        todo: reload old population

    """



    def __init__(self):
        pass



if __name__ == '__main__':
    testgene=gene.gene()
    testgene.load(filename='E:\\Documents\\Stage\\python\\models\\3geneWithVarProduction.model')
    testgene.absoluteTruth=testgene.solveODE()#fill it with and objective
    print(testgene.fitness())#should return perfect fitness!


    test=population()
    test.seed(testgene)
    test.maxTime=timedelta(minutes=5)
    test.maxGenerations=100
    test.evolve()

    test.population[0].displayResults()

    filename2='E:\\Documents\\Stage\\python\\models\\testmodel.model'
    test.population[0].save(filename2)

    #logfile
    logfile='E:\\Documents\\Stage\\python\\models\\logfile.txt'
    test.saveProgress(logfile)
    test.displayProgress()



