import sys
from gene import gene
from population import population
from datetime import timedelta
import copy

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

    # copy the individual
    newGene=copy.deepcopy(originalGene)

    # and randomize
    newGene.randomize()

    # local optimize
    newGene.localOptimize()

    # save
    newGene.save(dirName+os.sep+'optimized'+str(uniqueNumber)+'.model')

    # say good bye
    print('Done! HURRAY!')

