from datetime import datetime,timedelta
import math
import random
import scipy
from copy import deepcopy
from multiprocessing import Process,Pool
import os

class GA_population:
    """ GA_population genetic algoritm

        implementation of a genetic algorithm using GA_population with GA_gene to do magic!
        both need to be inherented so you can give your own implementations of mutation and crossover and fitness and stuff...
        but most of that is in the gene part....

        this genetic algorithm should give you:
        - progress loging
            - history ~ how did my gene evolve (far from perfect)
            - safe to a file
        - 3 stop conditions
            - x amount of time
            - x amount of generations
            - x amount of generations no progress
        - a very minimalistic implementation

        required that you fill an initial gene yourself... this gene then will then be used to create copies of it self which will be randomized by your implementation of the create function!

        has various settings... see code :P

        todo: threading/multiprocesses... or what's it called
        todo: purge individuals that have the same fitness... asume that they have the same variables....

        todo: smart cleansing? if population becomes to similar?

        note: history can become insanely large in long runs...
        todo: history toggle?

        this implementation is loosely based on the work of: http://freenet.mcnabhosting.com/python/pygene/

    """

    # Genetic Algoritm Settings
    cullToCount=20 # the size of the population
    incest=3 # number of surviving unchanged parents (1 or more best is always included rest is semi random)

    immigrants=5 # number of fully new randomly generated inidividuals (they will be added to the parents... before mutations and crossover...)
    childCount=5 # number of children generated (note that this is times 2)(crossover)
    mutantCount=10 # number of mutants generated

    # Stop conditions
    maxGenerations=None # the number of generations for the algorithm to run (None, ignores this as a stop condition)
    maxTime=None # max time as a datetime.timeDelta object!, if left empty it wont stop! (None, ignores this as a stop condition)
    maxGenerationWithSameFitness=25 # if you have x generation with no better fitness... then stop! (None, ignores this as a stop condition)

    # progress
    currentGeneration=0
    progressList=[] # table [generation, average fitness, lowest fitness, gene.ID, gene.history[see history in GA_gene]]
    printProgress=True # toggle if you want a progress report every generation printed to the console

    topFitness=None
    generationWithSameFitness=0
    previousLowestFitnessGeneHistory=None
    previousLowestFitnessValue=None


    # population
    population=[] # contains a number of GA_gene s ...

    def __init__(self):
        pass

    def seed(self, gene):
        """ seed the population using a single gene """
        print('Seeding initial population...')
##        # create a pool for multi threading
##        if os.name != 'nt':
##            pool=Pool()
##            for i in range(self.cullToCount-len(self.population)):
##                pool.apply_async(deepcopy(gene).__create__,callback=self.population.append)
##            pool.close()# make sure everything is done before continuing
##            pool.join()
##        else:
        for i in range(self.cullToCount-len(self.population)):
            self.population.append(deepcopy(gene).__create__())
        self.sort()

    def evolve(self):
        """ start the genetic algorithm """

        # time stuff
        checkTime=False
        startTime=datetime.now()
        if(self.maxTime!=None):
            endTime=startTime+self.maxTime
            checkTime=True
            print('startTime: '+str(startTime)+'\nplanned stoptime: '+str(endTime))

        # generation counter
        self.currentGeneration=0

        # check initial population
        if self.cullToCount>len(self.population):
            print('ERROR initial population to small!!')



        # you cannot stop evolution!
        while True:
            # make/import immigrants
##            print('imigrants')
            for i in range(self.immigrants):
                self.population.append(deepcopy(self.population[0]).__create__())
            self.sort()

            newPop=[]

            # do crossovers
##            print('crossovers')
            for i in range(self.childCount):
                newPop.extend(self.crossover())

            # make mutants
##            print('mutants')
            for i in range(self.mutantCount):
                newPop.append(self.mutate())

            # keep the parents
##            print('keep the parents')
            parentsToKeep=[]
            parentsToKeep.append(self.population[0]) # keep best parent
            self.population.pop(0) # dont want duplicates
            for i in range(self.incest-1): # keep some more parents
                # select a parent with a bias for the one with the highest fitness
                selector=(len(self.population)-1) - int(math.sqrt(random.uniform(0,(len(self.population))**2)))
                parentsToKeep.append(self.population[selector])
                self.population.pop(selector)


            # cull and switch
##            print('cull and switch')
            self.population=newPop # switch!
            self.sort() # sort cause of elimination bias
            self.cull # bring it down a notch
            self.population.extend(parentsToKeep) # dont forget about your roots!
            self.sort() #

            # update progress
            self.currentGeneration+=1
            self.updateProgress()



            # check stop conditions
            if self.topFitness==None: # fill it the first time
                self.topFitness=self.population[0]
            if self.population[0]==self.topFitness: # time with no progress limit
                self.generationWithSameFitness+=1
                if(self.maxGenerationWithSameFitness!=None and self.generationWithSameFitness>=self.maxGenerationWithSameFitness):
                    print('stopped due to a lack of progress')
                    s='startTime: '+str(startTime)+'\nstopTime: '+str(datetime.now())
                    print(s)
                    break # not making any progress

            else:
                self.generationWithSameFitness=0
                topFitness=self.population[0]

            if(checkTime and datetime.now()>=endTime): # time limit
                print('time limit reached')
                s='startTime: '+str(startTime)+'\nplannedStopTime:'+str(endTime)+'\nstopTime: '+str(datetime.now())
                print(s)
                break

            if(self.maxGenerations!=None and self.currentGeneration>=self.maxGenerations): # generation limit
                print('max number of generations reached')
                s='startTime: '+str(startTime)+'\nstopTime: '+str(datetime.now())
                print(s)
                break

            # add run until no change.... for a while...

    def cull(self):
        """ reduce the current population"""
        # must kill!!!
        for i in range(self.cullToCount-self.incest):
            selector=(len(self.population)-1) - int(math.sqrt(random.uniform(0,(len(self.population))**2)))
            self.population.pop(selector)

    def sort(self):
        """ sort the current population"""
##        # make sure all fitnesses are calculated
##        for gene in self.population:
##            gene.fitness()
##            print(gene.fitness())
        #sort
        self.population=sorted(self.population, key=lambda gene: gene.fitness())

    def updateProgress(self):
        """ update the progresslist, using the gene history, and print it to the console

            set printProgress=False to disable output to the console
        """
        # get the maximium and average fitness values
        lowestFitnessValue=None
        lowestFitnessGeneID=None
        lowestFitnessGeneHistory=None
        totalFitnessValue=0
        for gene in self.population:
            if(lowestFitnessValue==None): # first go!
                lowestFitnessValue=gene.fitness()
                lowestFitnessGeneID=gene.ID
                lowestFitnessGeneHistory=gene.history
            elif(gene.fitness<lowestFitnessValue): # lower fitness found! (better match!)
                lowestFitnessValue=gene.fitness()
                lowestFitnessGeneID=gene.ID
                lowestFitnessGeneHistory=gene.history
            else:
                pass # fitness value was higher... so ignore
            totalFitnessValue+=gene.fitness()

        #
        average=totalFitnessValue/len(self.population)
        #
        # put all the history in a nice array
        self.progressList.append([self.currentGeneration, average, lowestFitnessValue, lowestFitnessGeneID, lowestFitnessGeneHistory])

        # print progress
        if(self.printProgress):
            if(self.currentGeneration==1): # header
                print('generation - average fitness - lowest fitness - lowest fitness GeneID - (optional) lowest fitness gene\'s history')
            if(self.previousLowestFitnessGeneHistory==None or (lowestFitnessGeneID==self.previousLowestFitnessGeneHistory and lowestFitnessValue==self.previousLowestFitnessValue)):
                # same lowest or first
                s=str(self.currentGeneration) +'\t'+ str(average) +'\t'+ str(lowestFitnessValue) + '\t' + str(lowestFitnessGeneID)
            else:
                # new lowest
                s=str(self.currentGeneration) +'\t'+ str(average) +'\t'+ str(lowestFitnessValue) + '\t' + str(lowestFitnessGeneID)+ '\t' + str(lowestFitnessGeneHistory)
            print(s)

        #
        self.previousLowestFitnessGeneHistory=lowestFitnessGeneID
        self.previousLowestFitnessValue=lowestFitnessValue

    def saveProgress(self, filename):
        """ Store the progress report in a file"""
        f = open(filename, 'w')
        s=''
        for generation in self.progressList:
            for stuff in generation:
                s+=str(stuff)+'\t'
            s=s[:-1]+'\n' # remove extra tab and add end of line char
        f.write(s)
        f.close()

    def displayProgress(self):
        """display progress, requires pylab"""
        import pylab # put on this level as lisa has no pylab

        # having problems accessing these values with just [:][0:3]... so
        data=scipy.zeros([len(self.progressList),2])
        generations=scipy.zeros([len(self.progressList)])
        for i in range(len(self.progressList)):
            generations[i]=self.progressList[i][0]
            data[i,:]=self.progressList[i][1:3]
        pylab.plot(generations,data)
        pylab.show()

    def crossover(self):
        """select 2 random genes and do a crossover (exchange variables)"""
        # select random genes
        length=len(self.population)
        mummy=random.randint(0,length-1) # random selection
        daddy=(len(self.population)-1) - int(math.sqrt(random.uniform(0,(len(self.population))**2))) # biased selection

        while mummy == daddy:#make sure they are different
            daddy=(len(self.population)-1) - int(math.sqrt(random.uniform(0,(len(self.population))**2))) # biased selection

        # make copies
        mummy=deepcopy(self.population[mummy])
        daddy=deepcopy(self.population[daddy])

        # do the crossover
        mummy.__crossover__(daddy)

        return [mummy, daddy]


    def mutate(self):
        """select a random gene from population and mutate it"""
        # select a random gene and mutate it
        mutant=deepcopy(self.population[random.randint(0,len(self.population)-1)])
        mutant.__mutate__()
        return mutant


if __name__ == '__main__':
    pass

