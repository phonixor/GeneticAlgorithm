import sys
from gene import gene
from population import population
from datetime import timedelta

import argparse

##class experiment:
##    def __init__(self):
##        pass

if __name__ == '__main__':
    """ needs to be called form te command line!

        see experiment.sh

    """
##    print(sys.argv)
    doLocalOptimize=True

    parser = argparse.ArgumentParser(description='Run a genetic algorithm experiment!')
    parser.add_argument('dirName', help='dir in which the results of the genetic algorithm are stored, it should be path incl the new dir')
    parser.add_argument('originalGene', help='full path to the original gene file')
    parser.add_argument('-n',dest='uniqueNumber', help='unique number usefull for running lots of experiments at the same time, doesn\'t have to be a number')
    parser.add_argument('-l', dest='dontDoLocalOptimize', action='store_true', help='don do a local optimize... mostly for testing purposes only :P its an l not i')
##    parser.print_help()
    args=parser.parse_args()

    dirName=args.dirName
    originalGeneFile=args.originalGene

    if args.dontDoLocalOptimize:
        doLocalOptimize=False

    uniqueNumber=args.uniqueNumber


    print('------------------Settings-------------------')
    print('dirName : '+dirName)
    print('originalGeneFile : '+originalGeneFile)
    print('doLocalOptimize : '+str(doLocalOptimize))
    print('uniqueNumer : '+str(uniqueNumber))
    print('---------------------------------------------')


##    # get the command line parameters
##    dirname=sys.argv[0]
##    originalGeneFile=sys.argv[1]


    originalGene=gene()
    originalGene.load(originalGeneFile)
    goal=originalGene.solveODE()
    originalGene.absoluteTruth=goal


    # test fitness condition
    if originalGene.fitness() != 0:
        raise Exception('ERROR! this gene cannot predict its own ideal!!')

    # create the new dir
    import os
##    os.mkdir(dirName) # didnt work

    try:
        os.makedirs(dirName) # so lets try
        # make a copy of the origenal gene
        originalGene.save(dirName+os.sep+'originalGene.model')
    except:
        print('what went rong?')
        print(sys.exc_type)
        print(sys.exc_value)
        print(sys.exc_traceback)

    # start a new population
    pop=population()
    # set GA settings
    pop.maxTime=timedelta(hours=60)
    # create a population using the orignial gene
    pop.seed(originalGene)
    pop.evolve()

    #
    # make a models dir
##    os.mkdir(dirName+'models') # sometimes does not work on linux for some reason
    os.makedirs(dirName+os.sep+'models'+str(uniqueNumber))
    for i in range(len(pop.population)):
        pop.population[i].save(dirName+os.sep+'models'+str(uniqueNumber)+os.sep+'model_'+str(i)+'.model')

    # save history
    pop.saveProgress(dirName+os.sep+'log'+str(uniqueNumber)+'.txt')

    if doLocalOptimize:
        # do a local search for the best gene in the population
        pop.population[0].localOptimize()
        pop.population[0].save(dirName+os.sep+'optimized'+str(uniqueNumber)+'.model')

    print('Done! HURRAY!')

