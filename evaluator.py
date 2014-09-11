from gene import gene
from population import population

import tkFileDialog
from Tkinter import Tk

import scipy
import pylab
import copy
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons


class evaluator:
    """ used to evaluate populations and genes of the population and gene classes :P

        Requirements:
        works with the implementation gene.py of GA_gene.py
        uses:  fitness(), solveODE(), min max ranges, getVariableList().. etc
        ... so its not an automatic fit to GA_gene.py :(


        todo: better requirements
        todo history
        todo nice graphs
        todo informative
        todo profit
        todo remove the fitness 100000000


        todo: remove dependency on goal gene


        kinda done:
        - filter crappy out of bound solutions





    """
    population=[]
    goalGene=None
    absoluteTruth=None
##    misfits=[] # population out of bounds...
    removedPopulation=[]
    analysableVariables=[]

    def __init__(self):
        pass

    def openGoalFile(self, filename=None, normalize=True):
##        test=tkFileDialog
##        print(test)
##        print(type(test))
##        print(dir(test))
##        filename=test.askopenfilename()
        # might want to replace it with:
        # http://stackoverflow.com/questions/9319317/quick-and-easy-file-dialog-in-python


        if filename==None:
            # get the file name using a file dialog!
##            t=Tk()
##            t.withdraw()
            filename=tkFileDialog.askopenfilename()
            print(filename)
##            t.quit()

        self.goalGene=gene()
        self.goalGene.normalized=normalize
        self.goalGene.load(filename)

        self.absoluteTruth=self.goalGene.solveODE()
        self.goalGene.absoluteTruth=self.absoluteTruth

        if self.analysableVariables==[]:
            self.analysableVariables=scipy.array(range(len(self.goalGene.getVariableList())))

    def openFiles(self,filePaths=None):
        """ open files and adds them to the population, can be called multiple times...."""
        if self.absoluteTruth==None:
            raise Exception('first load original gene!!')

        if filePaths==None:
##            t=Tk()
##            t.withdraw()
            filePaths=tkFileDialog.askopenfilenames().split()
            print(filePaths)
##            t.quit()

##        print(type(filePaths))
##        print(filePaths)


        mi=self.goalGene.minParRangeList
        ma=self.goalGene.maxParRangeList

        nrOfMistFits=0


        for i in range(len(filePaths)):
            print('reading file '+str(i+1)+' of '+str(len(filePaths)))
            filename=filePaths[i]

            aGene=gene()
            aGene.normalized=self.goalGene.normalized
            aGene.load(filename)
            aGene.absoluteTruth=self.absoluteTruth


            # check if the individual is within the constraints

##            variables=aGene.getVariableList()
##            for i in range(len(mi)):
##                if variables[i]>ma[i] or variables[i]<mi[i]: # if out of bounds
##                    # put that individual in the trash bin
##                    self.misfits.append(aGene)
##            else:
##                aGene.fitness()
##                self.population.append(aGene)

##        print('nr of corrupted individuals:'+str(len(self.misfits)))
            isMisfit=False
            params=aGene.getVariableList()
            for i in range(len(mi)):
                # make sure it stays withing model constraints/bounds
                if params[i] < mi[i]:
                    print('correcting parameter from:' +str(params[i]) + ' to: '+ str(mi[i]))
                    params[i]=mi[i]
                    isMisfit=True

                if params[i] > ma[i]:
                    print('correcting parameter from:' +str(params[i]) + ' to: '+ str(ma[i]))
                    params[i]=ma[i]
                    isMisfit=True
            if isMisfit:
                nrOfMistFits+=1
            aGene.setVariableList(params)

            aGene.fitness()
            self.population.append(aGene)

        print('nr of corrupted individuals:'+str(nrOfMistFits))

        # sort it!
        self.sort()

    def savePopulation(self, dirname):
        pass

    def sort(self):
        self.population=sorted(self.population, key=lambda gene: gene.fitness())

    def removeTheWeak(self, minimumFitnessValue):
        """ removes all individuals with a lower fitness then the one given from the population

            the population is restored before each culling

            just give a very large number to restore the population :)

            todo: add commandline feedback # removed restored total
        """
        print('removing individuals with a fitness above : '+str(minimumFitnessValue) + ' from the population')

        # restore older removed population
        self.population.extend(self.removedPopulation)
        self.removedPopulation=[]
        i=0
        while True:
            if self.population[i].fitness()>minimumFitnessValue:
                self.removedPopulation.append(self.population[i])
                self.population.pop(i)
                # i remains the same :P
            else:
                i+=1
            # stop if needed
            if i==len(self.population):
                break
        self.sort()

    def cluster(self):
##        import scipy.cluster.hierarchy as magic
##        print('cluser magic!')
##
        variables=scipy.zeros([len(self.population),len(self.goalGene.getVariableList())])
        for i in range(len(self.population)):
            variables[i,:]=self.population[i].getVariableList()
##
##        link=magic.linkage(variables)
##
##        magic.dendrogram(link)
##        pylab.show()
        import prettyHierarchy
        prettyHierarchy.prettyHierachy(variables[:,self.analysableVariables])


    def showLogFiles(self,filePaths=None):
        if filePaths==None:
##            t=Tk()
##            t.withdraw()
##            Tk().withdraw()
            filePaths=tkFileDialog.askopenfilenames().split()
            print(filePaths)
##            t.quit()

        for i in range(len(filePaths)):
            print('reading file '+str(i+1)+' of '+str(len(filePaths)))
            filename=filePaths[i]

            averageFitnesses=[]
            bestFitnesses=[]



            f = open(filename, 'r')

            line=f.readline()

            while(line != ''): # if its an empty row an end of line char would be there
                line=line.strip().split("\t") #remove end of line / new line chars
                averageFitnesses.append(line[1])
                bestFitnesses.append(line[2])
                line=f.readline()

            f.close() # release the hostage

            plt1=pylab.plot(averageFitnesses, linestyle=':')
            plt2=pylab.plot(bestFitnesses)
            pylab.legend([plt1[0],plt2[0]],['average','best'],loc='upper right')

        pylab.title('Genetic algorithm progress')
        pylab.xlabel('generation')
        pylab.ylabel('fitness')
        pylab.show()

    def truthFinder(self, normalize=True):
        """
            shows how many times variables are predicted correctly when the fitness increases...
            this indicates how stable the variables are...

            todo filter always >0 cause they are always predicted correctly...

            NOT WORKING YET!!!


        """

        trueValues=scipy.array(self.goalGene.getVariableList())
        print('trueValues')
        print(trueValues)

        pnz=scipy.zeros(len(trueValues))
        pnz[trueValues>0]=1
        pnz[trueValues<0]=-1
        print('pnz')
        print(pnz)

        # create a single array containing all parameters of the population
        variables=scipy.zeros([len(self.population),len(self.goalGene.getVariableList())])
        for i in range(len(self.population)):
            variables[i,:]=self.population[i].getVariableList()



        correctlyPredictedPositive=scipy.zeros(variables.shape)
        correctlyPredictedNegative=scipy.zeros(variables.shape)

        predictedPositive=scipy.zeros(variables.shape)
        predictedNegative=scipy.zeros(variables.shape)
        predictedZero=scipy.zeros(variables.shape)
        totalPredictedPositive=scipy.zeros(variables.shape[1])
        totalPredictedPositiveSoFar=scipy.zeros(variables.shape)

        totalParameterValue=scipy.zeros(variables.shape[1])
        totalParameterValueSoFar=scipy.zeros(variables.shape)
        averageParameterValueSoFar=scipy.zeros(variables.shape)

        for i in range(len(self.population)):
##            print((trueValues>0).astype(int) * (variables[:,i]>0).astype(int)) # works
##            print(correctlyPredictedArray[((trueValues>0).astype(int) * (variables[:,i]>0).astype(int)).astype(bool),i])
            # compare the true values with the predicted values and put a 1 if they are correctly predicted positive or negative
            correctlyPredictedPositive[i,((trueValues>0).astype(int) * (variables[i,:]>0).astype(int)).astype(bool)]=1
            correctlyPredictedNegative[i,((trueValues<0).astype(int) * (variables[i,:]<0).astype(int)).astype(bool)]=1

            predictedPositive[i,(variables[i,:]>0).astype(int).astype(bool)]=1
            predictedNegative[i,(variables[i,:]<0).astype(int).astype(bool)]=1
            predictedZero[i,(variables[i,:]==0).astype(int).astype(bool)]=1
            #

            totalPredictedPositive[:]=totalPredictedPositive[:]+predictedPositive[i,:]
            totalPredictedPositiveSoFar[i,:]=totalPredictedPositive[:]


            totalParameterValue[:]=totalParameterValue[:]+variables[i,:]
            totalParameterValueSoFar[i,:]=totalParameterValue[:]
            averageParameterValueSoFar[i,:]=totalParameterValue[:]/(i+1)


        pylab.figure()
        pylab.plot(range(len(self.population)),totalPredictedPositiveSoFar[:,self.analysableVariables])
        pylab.legend(self.goalGene.variableNames[self.analysableVariables],loc='upper left')

        pylab.figure()
        pylab.plot(range(len(self.population)),averageParameterValueSoFar[:,self.analysableVariables])
        pylab.legend(self.goalGene.variableNames[self.analysableVariables],loc='upper right')


        pylab.show()



##
##
##        #show for each variable
##
##        pylab.subplots_adjust(left=0.25, bottom=0.25)
##
##        selection=scipy.ones(len(self.goalGene.geneNames)).astype(bool)# select all on start
##        plots = pylab.plot(self.results[:,-1,:]) # apperently returns a plot for each line...
##
##
##
##        axcolor = 'lightgoldenrodyellow'
##
##
##        def update(val):
##            for i in range(len(plots)):
##                plots[i].set_ydata(self.results[:,-1,i])
##                plots[i].set_visible(selection[i])
##
##            pylab.draw()
##
##
##        rax = pylab.axes([0.025, 0.5, 0.15, 0.15], axisbg=axcolor)
##
##
##
##        checker=CheckButtons(rax,self.geneNames,actives=scipy.ones(len(self.geneNames)))
##        def selector(val):
##
##            geneNr=scipy.array(range(len(self.geneNames)))[self.geneNames==val][0] # its retarded to check label names... but that is the way they like it....
##            selection[geneNr]=not(selection[geneNr])
##            update(slider.val)
##        checker.on_clicked(selector)
##        print(checker.eventson)
##        print(checker.drawon)
##
##
##        pylab.show()







##
##
##        pylab.plot(scipy.sum(correctlyPredictedArray,axis=0), color='red')
##        pylab.plot(averageNrOfTruthArray, color='blue')
##        pylab.show()




##
##        correctlyPredictedArray[variables[trueValues>0,:]]=1
##        correctlyPredictedArray[trueValues<0,:]=-1

##
##        for i in range(len(self.population)):
##            if trueValues[j]>0: # variable is positive
##                correctlyPredicted.append(sum(variables[j,:]>0))
##
##                pnz.append(1)
##            elif trueValues[j]<0: # variable is negative
##                correctlyPredicted.append(sum(variables[j,:]<0))
##
##                pnz.append(-1)
##            else: # variable = 0
##                correctlyPredicted.append(0)
##
##                pnz.append(0)




    def showVariables(self,normalize=True):
        """
            maybe add a normalize thing ... over total range... not really the data...

            i think this doesnt work without normalized stuff atm...
        """

        variables=scipy.zeros([len(self.goalGene.getVariableList()),len(self.population)])
        for i in range(len(self.population)):
            variables[:,i]=self.population[i].getVariableList()

        correctlyPredicted=[] # only has to predict if the variable is positive or negative
        bestCorrectlyPredicted=[]
        pnz=[]
        predictedPositive=[]
        predictedNegative=[]
        predictedZero=[]

        trueValues=scipy.array(self.goalGene.getVariableList())
##        print(variables.shape)
##        print(len(self.goalGene.getVariableList()))
##        print(range(1,len(self.goalGene.getVariableList())+1))
        if normalize: # weird place for this? prop need to be always on else nothing happends... mmmh...
            for i in self.analysableVariables: # for each variable
##                print(i)
                mi=self.goalGene.minParRangeList[i]
                ma=self.goalGene.maxParRangeList[i]

                if trueValues[i]>0: # variable is positive
                    correctlyPredicted.append(sum(variables[i,:]>0))
                    bestCorrectlyPredicted.append(variables[i,0]>0)
                    pnz.append(1)
                elif trueValues[i]<0: # variable is negative
                    correctlyPredicted.append(sum(variables[i,:]<0))
                    bestCorrectlyPredicted.append(variables[i,0]<0)
                    pnz.append(-1)
                else: # variable = 0
                    correctlyPredicted.append(0)
                    bestCorrectlyPredicted.append(0)
                    pnz.append(0)
                    # every thing is good... or everything is bad... but thats so negative...
                    # maybe only good if near 0.... but... meh...

                predictedNegative.append(sum(variables[i,:]<0))
                predictedPositive.append(sum(variables[i,:]>0))
                predictedZero.append(sum(variables[i,:]==0)) # should not happen very often...


                # want it in a range from 0-1
                addToGetMinToZero=0-mi
                fullrange=ma-mi

                # normalize
                variables[i,:]+=addToGetMinToZero
                variables[i,:]=variables[i,:]/fullrange

                trueValues[i]+=addToGetMinToZero
                trueValues[i]=trueValues[i]/fullrange



##        variables=scipy.array([1,2,3])
##        avarages=scipy.average(variables)
        avarages=scipy.apply_along_axis(scipy.average,1,variables[self.analysableVariables])
        bestValues=variables[self.analysableVariables,0] # take the first individual, assume that the population is sorted on fitness
        correctlyPredicted=scipy.array(correctlyPredicted).astype(float)
        bestCorrectlyPredicted=scipy.array(bestCorrectlyPredicted).astype(float)

        correctyPredictedPercentage=((correctlyPredicted/len(self.population))*100).astype(int)
        correctyPredictedPercentageGT50=(sum(correctyPredictedPercentage[correctyPredictedPercentage>50])/len(correctyPredictedPercentage))*100

        print('--------------------------------------------------------------------------')
        print('nr of individuals    : '+str(len(self.population)))
        print('--------------------------------------------------------------------------')
        print('true variables       : '+str(self.goalGene.getVariableList()))
        print('avarage variables    : '+str(avarages))
        print('best variables       : '+str(bestValues))
        print('positive/negative/0  : '+str(pnz))
        print('# predicted negative : '+str(predictedNegative))
        print('# predicted positive : '+str(predictedPositive))
        print('# predicted zero     : '+str(predictedZero))
        print('correctly predicted #: '+str(correctlyPredicted))
##        correctlyPredicted[correctlyPredicted==-1]=0 # set it so that 0 is set to 0%... else its just confusing...

        print('correctly predicted %: '+str(correctyPredictedPercentage))
        print('best predicted       : '+str(bestCorrectlyPredicted))
        print('--------------------------------------------------------------------------')


        print('average predicted >50% correct % :'+str(correctyPredictedPercentageGT50))
        print('best predicted %     : '+str((sum(bestCorrectlyPredicted)/len(bestCorrectlyPredicted))*100)) # also counts zeros... which aint funny



##        x=range(1,len(self.goalGene.getVariableList())+1)
        x=self.analysableVariables
        p1=pylab.plot(x, variables[self.analysableVariables], linewidth=0, marker='.', color='#cccccc', markeredgecolor='#cccccc')# cccccc=ligth gray
        pylab.xlim(xmin=-1,xmax=len(self.analysableVariables))
        p2=pylab.plot(x, avarages, color='yellow')
        p3=pylab.plot(x, bestValues, color='red' )
        p4=pylab.plot(x, trueValues[self.analysableVariables], color='blue')
        pylab.legend((p1[0],p2[0],p3[0],p4[0]),('values','average','best predicted','actual'))
        print(self.goalGene.variableNames)
        pylab.xticks(self.analysableVariables,self.goalGene.variableNames[self.analysableVariables], rotation=45)
##        pylab.xlabel('variables')
        pylab.ylabel('normalized values')
        pylab.title('variables')
        pylab.show()


    def showHistory(self):
        """ show how much history the current population has
            to test if its global algorithm magic... or just luck!
        """
        mutations=scipy.zeros(len(self.population))
        crossovers=scipy.zeros(len(self.population))
        other=scipy.zeros(len(self.population)) # create and local optimize
        fitnesses=[]
        tempFitness=[]
        for i in range(len(self.population)):
            history=self.population[i].history
            for j in range(len(history)): # for each row in history check
                #
                tempFitness.append(history[j][1])
                #
                if history[j][0]=='mutate':
                    mutations[i]+=1
                elif history[j][0]=='crossover':
                    crossovers[i]+=1
                else:
                   other[i]+=1
            #
            fitnesses.append(tempFitness)
            tempFitness=[]
##        print(mutations)
##        print(crossovers)
##        print(other)


        # plot
        x=range(len(self.population))

        p1=pylab.bar(x,other, color='red')
        p2=pylab.bar(x,crossovers, color='green', bottom=other)
        bottom2=other+crossovers
        p3=pylab.bar(x,mutations, color='yellow', bottom=bottom2)

        pylab.ylabel('#')
        pylab.xlabel('solutions sorted by fitness (best is on the left)')
        pylab.title('genetic algorithm actions')
        pylab.legend((p1[0],p2[0],p3[0]),('rest','crossovers','mutations'))


##        pylab.plot(x,mutations,x,crossovers,x,other)
##        pylab.bar(x,mutations,x,crossovers,x,other)







        # calculate how well the algoritms did
        begin=scipy.zeros(len(self.population))
        beforeLocal=scipy.zeros(len(self.population))
        last=scipy.zeros(len(self.population))
        localsImprove=scipy.zeros(len(self.population))
##        globalImprove=scipy.zeros(len(self.population))
        partLocalImprove=scipy.zeros(len(self.population))

        for i in range(len(self.population)):
            history=self.population[i].history
            if history[-1][0] != 'localOptimize' :
                print('no local optimize!!!')
            begin[i]=history[0][1]
            beforeLocal[i]=history[-2][1]
            last[i]=history[-1][1]

            total=begin[i]-last[i]
            localsImprove[i]=beforeLocal[i]-last[i]
            partLocalImprove[i]=localsImprove[i]/total








        plotStuff=scipy.zeros([3,len(self.population)])
        plotStuff[0,:]=scipy.array(begin)
        plotStuff[1,:]=scipy.array(beforeLocal)
        plotStuff[2,:]=scipy.array(last)


##        print(plotStuff)

        fig2=pylab.figure()
        pylab.plot(plotStuff)


        print('partLocalImprove')
        print(partLocalImprove)


        averageImproved=sum(partLocalImprove)/len(self.population)
        print('partLocalImproveAvarage')
        print(averageImproved)
        print('partLocalImproveSD')
        print(scipy.std(partLocalImprove))






        #
        pylab.figure()
##        print('---------------------------------------')
##        print(fitnesses)
        for i in range(len(fitnesses)):
##            print(fitnesses[i])
            pylab.plot(fitnesses[i])
        pylab.show()


    def parameterSensitivityAnalysis(self, individual):
        """ ignores bounds
            might want ot remove the self thing.... but meh...
        """
        # get the stuff
        values=individual.getVariableList()
        mi=individual.minParRangeList
        ma=individual.maxParRangeList


        # for each parameter create a testset
        testSet=scipy.zeros([len(values),7])
        for i in range(len(values)):
            mami=ma[i]-mi[i]
            testSet[i,0]=values[i]-0.1*mami # 10% decrease
            testSet[i,1]=values[i]-0.05*mami # 5% decrease
            testSet[i,2]=values[i]-0.01*mami # 1% decrease
            testSet[i,3]=values[i] # 100%
            testSet[i,4]=values[i]+0.01*mami # 1% increase
            testSet[i,5]=values[i]+0.05*mami # 5% increase
            testSet[i,6]=values[i]+0.1*mami # 10% increase

        # for each test in the testset... do the test
        result=scipy.zeros(testSet.shape)
        for i in range(testSet.shape[0]):
            print('variable '+str(i+1)+' of ' + str(len(values)))
            for j in range(testSet.shape[1]):
                # edit the variable list
                tempValues=copy.copy(values) # restore to indevidual # damn copy was needed here!
                tempValues[i]=testSet[i,j] #do the mutation
                # calculate the fitness
                result[i,j]=individual.fitness_ForLocalOptimize(tempValues)

        # restore to individuals default
        individual.setVariableList(values)

        # determine maximum value so that all figures are scaled equally
        ymax=result.max(axis=0).max(axis=0)
        if ymax>1: # prevent a few big ones ruin it for the rest...
            ymax=1

        # display the result
        remainingPop=len(values)
        offset=0

        while(remainingPop>0):# for each variable
            # create figures of 12 pictures at a time...
            # always starting with the real data!

            # decide nr of plots
            nrOfPlots=remainingPop
            if(nrOfPlots>12):
                nrOfPlots=12

            pylab.figure()
            # create the rest of the grid of plots
            for i in range(nrOfPlots):

                pylab.subplot(3,4,i+1)
                pylab.plot(result[offset+i,:])
                pylab.title(individual.variableNames[offset+i])
                print(testSet[offset+i,:])
##                pylab.xticks(testSet[offset+i,:])
                pylab.xticks(range(7),testSet[offset+i,:])
##                pylab.xlim()
##                pylab.ylim(ymin=0, ymax=ymax) # this one is bad if you have a single fitness that ruines everythinng!
                pylab.ylim(ymin=0, ymax=ymax)


            # update variables
            remainingPop-=nrOfPlots
            offset+=nrOfPlots



        pylab.show()

    def evaluateSolutionStability(self, nrOfSolutionsUsed=3):
        """ using parameter sensitivity it is determined how
            if a solution is stable...

            currently fluxuates by 1%.... that might be a to small fluxuation...

            set analysableVariables to filter the stuff you want

            todo: only variable interesting switch

            BUGGED!!! every image is the same!!!!

        """

        # get the best individuals
        if nrOfSolutionsUsed > len(self.population): # make sure you have that many
            print(str('ERROR: Not that many individuals, aksed for: ',nrOfSolutionsUsed,' but the population is only: '+len(self.population)+' individuals.' ))
        bestIndividuals=copy.deepcopy(self.population[0:nrOfSolutionsUsed])# deep copy to make sure they aren't changed

        # get the stuff
        individual=bestIndividuals[0]
        mi=individual.minParRangeList
        ma=individual.maxParRangeList


        # for each individual * each parameter create a testset
        print('creating testset')
        testSet=scipy.zeros([nrOfSolutionsUsed,len(mi),3]) # [nrOfIndividuals,nrOfVariables, theSpecificMutation] ... maybe anylsisblvrarbles... here...
        for i in range(nrOfSolutionsUsed): # for each individual
            values=bestIndividuals[i].getVariableList() # get variable list
##            for j in range(len(values)): # for each variable in that list, do 2 permutations
            for j in self.analysableVariables:
                mami=ma[j]-mi[j] # all models are/should be the same
                testSet[i,j,0]=values[j]-0.01*mami # 1% decrease
                testSet[i,j,1]=values[j] # 100%
                testSet[i,j,2]=values[j]+0.01*mami # 1% increase

        print(testSet)

        # for each test in the testset... do the test
        print('running testset')
        result=scipy.zeros(testSet.shape)
        for i in range(testSet.shape[0]): # for each individual
            print('working on individual '+str(i+1)+' of '+str(testSet.shape[0]))
            values=bestIndividuals[i].getVariableList()# need to get the correct models variables
##            for j in range(testSet.shape[1]): # for each variable
            for j in self.analysableVariables:
                for k in range(testSet.shape[2]): # the variance in that variable
                    tempValues=copy.copy(values) # restore to indevidual # damn copy was needed here!
##                    tempValues[k]=testSet[i,j,k] # replace it with the test value -- wrong!!! its the variable i need to replace not 0,1,2...
                    tempValues[j]=testSet[i,j,k] # replace it with the test value
                    result[i,j,k]=individual.fitness_ForLocalOptimize(tempValues) # run the model

        # so now 2 3D arrays with:
        # [individual, variable, mutation] - changed variable
        # [individual, variable, mutation] - fitness


        # we are interested in variables now, and how stable they are over the population...
        # if they fluxuate a lot but those fluxuations have little impact on the fitness, than maybe its not that important...

        print('processing/evaluating results')

        differencePositive=scipy.zeros([result.shape[1],result.shape[0]])
        differenceNegative=scipy.zeros([result.shape[1],result.shape[0]])
        totalDiffernecePositive=scipy.zeros([result.shape[1]])
        totalDifferneceNegative=scipy.zeros([result.shape[1]])


##        for i  in range(result.shape[1]): # for each variable
        for i in self.analysableVariables:
            for j in range(result.shape[0]): # from each individual
                # this part cant really be generic...
                differencePositive[i,j]=((result[j,i,0]-result[j,i,1])/result[j,i,1])
                differenceNegative[i,j]=((result[j,i,2]-result[j,i,1])/result[j,i,1])
                #
                #
                totalDiffernecePositive[i]+=differencePositive[i,j]
                totalDifferneceNegative[i]+=differenceNegative[i,j]
            #
            #print('variable: '+ str(i)+' -'+str(totalDifferneceNegative[i])+' +'+str(totalDiffernecePositive[i]) )

##            pylab.figure()
##            pylab.plot(differenceNegative[i,:])
        #
##        pylab.show()

        totalText='global result:' + str(self.population[0].fitness())+ " min: "+str(result.min(axis=0).min(axis=0).min(axis=0))+" max:" + str(result.max(axis=0).max(axis=0).max(axis=0))

        # displaying results:
##        for i in range(result.shape[1]): # for each variable
        for i in self.analysableVariables:
            title=self.goalGene.variableNames[i]+' -'+str(totalDifferneceNegative[i])+' +'+str(totalDiffernecePositive[i])
            print(title)
            print('    negative    original     positive     ')


            #title max hight... etc...
            fig=pylab.figure()
##            pylab.title(title)

            # add variable values and variable range... values as subplot titles?

            fig.text(0.1,0.95,title)
            fig.text(0.1,0.9,totalText)
##            fig.text(0.1,0.3,variableText)
            fig.subplots_adjust(top=0.85) # make room for text

##            print(result[:,i,:])
##            print(result[:,i,:].max(axis=0))
##            print(result[:,i,:].max(axis=0).max(axis=0))

            # ymin and max based on the max/min of that variable only
##            ymin=result[:,i,:].min(axis=0).min(axis=0)
##            ymax=result[:,i,:].max(axis=0).max(axis=0)
            # this might be nicer... as it shows more impact...
            ymin=0
            ymax=result[:,:,:].max(axis=0).max(axis=0).max(axis=0)


            for j in range(result.shape[0]): # per individual
                print(str(j)+': '+str(result[j,i,0]) + ' ' + str(result[j,i,1])+' '+str(result[j,i,2]))
                ax=fig.add_subplot(1,result.shape[0],j)

                pylab.ylim(ymin,ymax) #why cant this work on the fucking plot! :@
                x=[1,2,3]
                x=[testSet[j,i,0],testSet[j,i,1],testSet[j,i,2]]
                pylab.plot(x,[result[j,i,0], result[j,i,1], result[j,i,2]] )
                pylab.xticks(x)



        pylab.show()
        #...


##        pylab.figure()
##        pylab.plot(testSet)
##        pylab.show()
    def showTimeGrid(self, setNormalizeTo=None):
        """
            5xtime series of 5
            first 5 is goalGene

            setNormalizeTo -- to force the population
        """
        import numpy

        nrOfRows=7
        nrOfColumns=5


        # backup population so that the real population isn't modified
        backupPopulation=copy.deepcopy(self.population)# cause setNormalizeTo could change population
        backupGoal=copy.deepcopy(self.goalGene)

        if setNormalizeTo!=None:
            self.goalGene.normalized=setNormalizeTo
            self.goalGene.solveODE() # recalculate

            for i in range(0,nrOfRows-1):
                backupPopulation[i].normalized=setNormalizeTo
                backupPopulation[i].solveODE()


        timepoints=numpy.linspace(0,self.goalGene.endTime,5)
        print(timepoints)



        fig = pylab.figure(figsize=(8,8))
##        title = fig.add_axes([0.09,0.1,0.2,0.6])
##        pylab.text(.9,.9,'original model')




        xSize=0.8/nrOfColumns
        ySize=0.8/nrOfRows


        print(range(0,nrOfRows))
        for h in range(0,nrOfRows):# row
            for i in range(0,nrOfColumns): # column
                fig.add_axes([(0.05+(i*xSize)+(0.02*i)),(1-ySize)-(0.05+(h*ySize)+(0.02*h)),xSize,ySize])
##                pylab.subplot(nrOfRows,5,i+1+(h*5))

                if h==0: # first do the original model
                    pylab.plot(self.goalGene.results[:,i*25,:]) # the 25 is because of 101 time points
                    pylab.title('time: '+str(timepoints[i]), fontsize=10)
                else:
                    pylab.plot(backupPopulation[h-1].results[:,i*25,:])
                pylab.xticks([50],[""])
                pylab.yticks(fontsize=10)
            #
            # add label
##            fig.add_axes([.8,(1-ySize)-(0.05+(h*ySize)+(0.02*h)),xSize,ySize])
            if h==0:

                pylab.text(105,0.9,'original model', rotation=270)

            else:
                text=''+str(round(float(backupPopulation[h-1].history[len(backupPopulation[h-1].history)-1][1]),4))
##                print(backupPopulation[h-1].history)
                # +round(,5)
                pylab.text(105,0.9,text, rotation=270) # the 105 shouldn't realy be hardcoded.. but meh...


        pylab.show()







##          old code... without axes/axis/... whatever...
##
##
##        timepoints=numpy.linspace(0,self.goalGene.endTime,5)
##        print(timepoints)
##        pylab.figure()
##        pylab.text(1,1,'original model')
##        print(range(0,nrOfRows))
##        for h in range(0,nrOfRows):# row
##            for i in range(0,5): # column
##                pylab.subplot(nrOfRows,5,i+1+(h*5))
##                if h==0: # first do the original model
##                    pylab.plot(self.goalGene.results[:,i*25,:]) # the 25 is because of 101 time points
##                    pylab.title('time: '+str(timepoints[i]), fontsize=10)
##                else:
##                    pylab.plot(backupPopulation[h-1].results[:,i*25,:])
##                pylab.xticks([50],[""])
##                pylab.yticks(fontsize=10)
##
##
##
##        pylab.show()

        # restore goalGene
        self.goalGene=backupGoal




    def showFinalValuesGrid(self):
        """ create a picture grid for the entire population

            this is handy for deciding what a nice cutoff fitness value is

        """



        remainingPop=len(self.population)
        offset=0

        while(remainingPop>0):
            # create figures of 12 pictures at a time...
            # always starting with the real data!

            # decide nr of plots
            nrOfPlots=remainingPop
            if(nrOfPlots>11):
                nrOfPlots=11

            # new figure
            pylab.figure()

            # show the real data
            pylab.subplot(3,4,1)
            pylab.plot(self.goalGene.results[:,-1,:])
            pylab.xticks([0],[""])
            pylab.title('original data')


            # create the rest of the grid of plots
            for i in range(nrOfPlots):

                pylab.subplot(3,4,i+2)
                pylab.plot(self.population[offset+i].results[:,-1,:])
                pylab.xticks([0],[""])
                pylab.title('#'+str(offset+i)+' : '+str(self.population[offset+i].fitness()))




            # update variables
            remainingPop-=nrOfPlots
            offset+=nrOfPlots

        pylab.show()

    def showHierarchy(self):
        """ """

        # get data
        data=scipy.zeros([len(self.population),len(self.analysableVariables)])
        fitnessValues=scipy.zeros(len(self.population))
        for i in range(len(self.population)):
            data[i,:]=scipy.array(self.population[i].getVariableList())[self.analysableVariables]
            fitnessValues[i]=self.population[i].fitness()

##        print('test')
##        print(data.shape)
##        print(data[0:90,:].shape)
##        print(fitnessValues[0:90].shape)

        # show the data
        import prettyHierarchy
##        prettyHierarchy.prettyHierachy(data, ylabels=fitnessValues)
        prettyHierarchy.prettyHierachy(data[0:90,:], ylabels=fitnessValues[0:90]) # first 90 or 91.. only.... else it wont fit my damn screen!!!
        #i get a: valueError: Length n of condensed distance matrix 'y' must be a binomial


    def showPCA(self):
        """doesn't show anything yet...  """
        # maybe add fitness
        from matplotlib.mlab import PCA

        # get all variables
        variableArray=scipy.zeros([len(self.population),len(self.goalGene.getVariableList())])
        for i in range(len(self.population)):
            variableArray[i,:]=self.population[i].getVariableList()

        pcaResult=PCA(variableArray)

        print('mu')
        print(pcaResult.mu)
        print('sigma')
        print(pcaResult.sigma)
        print('y')
        print(pcaResult.Y)
        print('a')
        print(pcaResult.a)
        print('fracs')
        print(pcaResult.fracs)
        print('wt')
        print(pcaResult.Wt)



if __name__ == '__main__':
    e=evaluator()

##    e.showLogFiles()

##    goal='D:/Documents/Stage/python/models/test1/models/test1/originalGene.model'
##    goal=None
##    goal='D:/Documents/Stage/python/models/3geneWithVarProduction.model'
    goal='D:/Documents/Stage/python/models/4geneWithVarProduction.model'
    e.openGoalFile(goal)
##    e.openGoalFile(goal,normalize=False)

##    e.showLogFiles()
##    e.parameterSensitivityAnalysis(e.goalGene)
##    e.analysableVariables=scipy.array(range(9,18)) # the last 9 contains the values of the interaction matrix 3gene only of course...
    e.analysableVariables=scipy.array(range(12,28))
##    print(e.analysableVariables)
##

##
####    rest=[u'D:/Documents/Stage/python/models/test1/models/test1/models/model_0.model', u'D:/Documents/Stage/python/models/test1/models/test1/models/model_1.model', u'D:/Documents/Stage/python/models/test1/models/test1/models/model_2.model', u'D:/Documents/Stage/python/models/test1/models/test1/models/model_3.model', u'D:/Documents/Stage/python/models/test1/models/test1/models/model_4.model', u'D:/Documents/Stage/python/models/test1/models/test1/models/model_5.model', u'D:/Documents/Stage/python/models/test1/models/test1/models/model_6.model', u'D:/Documents/Stage/python/models/test1/models/test1/models/model_7.model', u'D:/Documents/Stage/python/models/test1/models/test1/models/model_8.model', u'D:/Documents/Stage/python/models/test1/models/test1/models/model_9.model', u'D:/Documents/Stage/python/models/test1/models/test1/models/model_10.model', u'D:/Documents/Stage/python/models/test1/models/test1/models/model_11.model', u'D:/Documents/Stage/python/models/test1/models/test1/models/model_12.model', u'D:/Documents/Stage/python/models/test1/models/test1/models/model_13.model', u'D:/Documents/Stage/python/models/test1/models/test1/models/model_14.model', u'D:/Documents/Stage/python/models/test1/models/test1/models/model_15.model', u'D:/Documents/Stage/python/models/test1/models/test1/models/model_16.model', u'D:/Documents/Stage/python/models/test1/models/test1/models/model_17.model', u'D:/Documents/Stage/python/models/test1/models/test1/models/model_18.model', u'D:/Documents/Stage/python/models/test1/models/test1/models/model_19.model', u'D:/Documents/Stage/python/models/test1/models/test1/models/model_20.model', u'D:/Documents/Stage/python/models/test1/models/test1/models/model_21.model', u'D:/Documents/Stage/python/models/test1/models/test1/models/model_22.model']
    rest=None
##    rest=[u'D:/Documents/Stage/python/models/test1_2/optimized1.model', u'D:/Documents/Stage/python/models/test1_2/optimized2.model', u'D:/Documents/Stage/python/models/test1_2/optimized3.model', u'D:/Documents/Stage/python/models/test1_2/optimized4.model', u'D:/Documents/Stage/python/models/test1_2/optimized5.model', u'D:/Documents/Stage/python/models/test1_2/optimized6.model', u'D:/Documents/Stage/python/models/test1_2/optimized7.model', u'D:/Documents/Stage/python/models/test1_2/optimized8.model', u'D:/Documents/Stage/python/models/test1_2/optimized9.model', u'D:/Documents/Stage/python/models/test1_2/optimized10.model', u'D:/Documents/Stage/python/models/test1_2/optimized11.model', u'D:/Documents/Stage/python/models/test1_2/optimized12.model', u'D:/Documents/Stage/python/models/test1_2/optimized13.model', u'D:/Documents/Stage/python/models/test1_2/optimized14.model', u'D:/Documents/Stage/python/models/test1_2/optimized15.model', u'D:/Documents/Stage/python/models/test1_2/optimized16.model',u'D:/Documents/Stage/python/models/test1_2/optimized17.model', u'D:/Documents/Stage/python/models/test1_2/optimized19.model', u'D:/Documents/Stage/python/models/test1_2/optimized20.model', u'D:/Documents/Stage/python/models/test1_2/optimized21.model', u'D:/Documents/Stage/python/models/test1_2/optimized22.model', u'D:/Documents/Stage/python/models/test1_2/optimized23.model', u'D:/Documents/Stage/python/models/test1_2/optimized24.model', u'D:/Documents/Stage/python/models/test1_2/optimized25.model', u'D:/Documents/Stage/python/models/test1_2/optimized26.model', u'D:/Documents/Stage/python/models/test1_2/optimized27.model', u'D:/Documents/Stage/python/models/test1_2/optimized28.model', u'D:/Documents/Stage/python/models/test1_2/optimized29.model', u'D:/Documents/Stage/python/models/test1_2/optimized30.model', u'D:/Documents/Stage/python/models/test1_2/optimized31.model', u'D:/Documents/Stage/python/models/test1_2/optimized32.model', u'D:/Documents/Stage/python/models/test1_2/optimized33.model', u'D:/Documents/Stage/python/models/test1_2/optimized34.model', u'D:/Documents/Stage/python/models/test1_2/optimized35.model', u'D:/Documents/Stage/python/models/test1_2/optimized36.model', u'D:/Documents/Stage/python/models/test1_2/optimized37.model', u'D:/Documents/Stage/python/models/test1_2/optimized38.model', u'D:/Documents/Stage/python/models/test1_2/optimized39.model', u'D:/Documents/Stage/python/models/test1_2/optimized40.model', u'D:/Documents/Stage/python/models/test1_2/optimized41.model', u'D:/Documents/Stage/python/models/test1_2/optimized42.model', u'D:/Documents/Stage/python/models/test1_2/optimized43.model', u'D:/Documents/Stage/python/models/test1_2/optimized44.model', u'D:/Documents/Stage/python/models/test1_2/optimized45.model', u'D:/Documents/Stage/python/models/test1_2/optimized46.model', u'D:/Documents/Stage/python/models/test1_2/optimized47.model', u'D:/Documents/Stage/python/models/test1_2/optimized48.model', u'D:/Documents/Stage/python/models/test1_2/optimized49.model', u'D:/Documents/Stage/python/models/test1_2/optimized50.model', u'D:/Documents/Stage/python/models/test1_2/optimized51.model', u'D:/Documents/Stage/python/models/test1_2/optimized52.model', u'D:/Documents/Stage/python/models/test1_2/optimized53.model', u'D:/Documents/Stage/python/models/test1_2/optimized54.model', u'D:/Documents/Stage/python/models/test1_2/optimized55.model', u'D:/Documents/Stage/python/models/test1_2/optimized56.model', u'D:/Documents/Stage/python/models/test1_2/optimized57.model', u'D:/Documents/Stage/python/models/test1_2/optimized58.model', u'D:/Documents/Stage/python/models/test1_2/optimized59.model', u'D:/Documents/Stage/python/models/test1_2/optimized60.model', u'D:/Documents/Stage/python/models/test1_2/optimized61.model', u'D:/Documents/Stage/python/models/test1_2/optimized62.model', u'D:/Documents/Stage/python/models/test1_2/optimized63.model', u'D:/Documents/Stage/python/models/test1_2/optimized64.model', u'D:/Documents/Stage/python/models/test1_2/optimized65.model', u'D:/Documents/Stage/python/models/test1_2/optimized66.model', u'D:/Documents/Stage/python/models/test1_2/optimized67.model', u'D:/Documents/Stage/python/models/test1_2/optimized68.model', u'D:/Documents/Stage/python/models/test1_2/optimized69.model', u'D:/Documents/Stage/python/models/test1_2/optimized70.model', u'D:/Documents/Stage/python/models/test1_2/optimized71.model', u'D:/Documents/Stage/python/models/test1_2/optimized72.model', u'D:/Documents/Stage/python/models/test1_2/optimized73.model', u'D:/Documents/Stage/python/models/test1_2/optimized74.model', u'D:/Documents/Stage/python/models/test1_2/optimized75.model', u'D:/Documents/Stage/python/models/test1_2/optimized76.model', u'D:/Documents/Stage/python/models/test1_2/optimized77.model', u'D:/Documents/Stage/python/models/test1_2/optimized78.model', u'D:/Documents/Stage/python/models/test1_2/optimized79.model', u'D:/Documents/Stage/python/models/test1_2/optimized80.model', u'D:/Documents/Stage/python/models/test1_2/optimized81.model', u'D:/Documents/Stage/python/models/test1_2/optimized82.model', u'D:/Documents/Stage/python/models/test1_2/optimized83.model', u'D:/Documents/Stage/python/models/test1_2/optimized84.model', u'D:/Documents/Stage/python/models/test1_2/optimized85.model', u'D:/Documents/Stage/python/models/test1_2/optimized86.model', u'D:/Documents/Stage/python/models/test1_2/optimized87.model', u'D:/Documents/Stage/python/models/test1_2/optimized88.model', u'D:/Documents/Stage/python/models/test1_2/optimized89.model', u'D:/Documents/Stage/python/models/test1_2/optimized90.model', u'D:/Documents/Stage/python/models/test1_2/optimized91.model', u'D:/Documents/Stage/python/models/test1_2/optimized92.model', u'D:/Documents/Stage/python/models/test1_2/optimized93.model', u'D:/Documents/Stage/python/models/test1_2/optimized94.model', u'D:/Documents/Stage/python/models/test1_2/optimized95.model', u'D:/Documents/Stage/python/models/test1_2/optimized96.model', u'D:/Documents/Stage/python/models/test1_2/optimized97.model', u'D:/Documents/Stage/python/models/test1_2/optimized98.model', u'D:/Documents/Stage/python/models/test1_2/optimized99.model', u'D:/Documents/Stage/python/models/test1_2/optimized100.model']
####    rest=[u'D:/Documents/Stage/python/models/test2/test2optimized1.model', u'D:/Documents/Stage/python/models/test2/test2optimized2.model', u'D:/Documents/Stage/python/models/test2/test2optimized3.model', u'D:/Documents/Stage/python/models/test2/test2optimized4.model', u'D:/Documents/Stage/python/models/test2/test2optimized5.model', u'D:/Documents/Stage/python/models/test2/test2optimized6.model', u'D:/Documents/Stage/python/models/test2/test2optimized7.model', u'D:/Documents/Stage/python/models/test2/test2optimized8.model', u'D:/Documents/Stage/python/models/test2/test2optimized9.model', u'D:/Documents/Stage/python/models/test2/test2optimized10.model', u'D:/Documents/Stage/python/models/test2/test2optimized11.model', u'D:/Documents/Stage/python/models/test2/test2optimized12.model', u'D:/Documents/Stage/python/models/test2/test2optimized13.model', u'D:/Documents/Stage/python/models/test2/test2optimized14.model', u'D:/Documents/Stage/python/models/test2/test2optimized15.model', u'D:/Documents/Stage/python/models/test2/test2optimized16.model', u'D:/Documents/Stage/python/models/test2/test2optimized17.model', u'D:/Documents/Stage/python/models/test2/test2optimized18.model', u'D:/Documents/Stage/python/models/test2/test2optimized19.model', u'D:/Documents/Stage/python/models/test2/test2optimized20.model', u'D:/Documents/Stage/python/models/test2/test2optimized21.model', u'D:/Documents/Stage/python/models/test2/test2optimized22.model', u'D:/Documents/Stage/python/models/test2/test2optimized23.model', u'D:/Documents/Stage/python/models/test2/test2optimized24.model', u'D:/Documents/Stage/python/models/test2/test2optimized26.model', u'D:/Documents/Stage/python/models/test2/test2optimized27.model', u'D:/Documents/Stage/python/models/test2/test2optimized28.model', u'D:/Documents/Stage/python/models/test2/test2optimized29.model', u'D:/Documents/Stage/python/models/test2/test2optimized30.model', u'D:/Documents/Stage/python/models/test2/test2optimized32.model', u'D:/Documents/Stage/python/models/test2/test2optimized34.model', u'D:/Documents/Stage/python/models/test2/test2optimized35.model', u'D:/Documents/Stage/python/models/test2/test2optimized36.model', u'D:/Documents/Stage/python/models/test2/test2optimized37.model', u'D:/Documents/Stage/python/models/test2/test2optimized39.model', u'D:/Documents/Stage/python/models/test2/test2optimized42.model', u'D:/Documents/Stage/python/models/test2/test2optimized43.model', u'D:/Documents/Stage/python/models/test2/test2optimized44.model', u'D:/Documents/Stage/python/models/test2/test2optimized45.model', u'D:/Documents/Stage/python/models/test2/test2optimized47.model', u'D:/Documents/Stage/python/models/test2/test2optimized48.model', u'D:/Documents/Stage/python/models/test2/test2optimized49.model', u'D:/Documents/Stage/python/models/test2/test2optimized50.model', u'D:/Documents/Stage/python/models/test2/test2optimized51.model', u'D:/Documents/Stage/python/models/test2/test2optimized52.model', u'D:/Documents/Stage/python/models/test2/test2optimized53.model', u'D:/Documents/Stage/python/models/test2/test2optimized54.model', u'D:/Documents/Stage/python/models/test2/test2optimized55.model', u'D:/Documents/Stage/python/models/test2/test2optimized58.model', u'D:/Documents/Stage/python/models/test2/test2optimized59.model', u'D:/Documents/Stage/python/models/test2/test2optimized60.model', u'D:/Documents/Stage/python/models/test2/test2optimized61.model', u'D:/Documents/Stage/python/models/test2/test2optimized62.model', u'D:/Documents/Stage/python/models/test2/test2optimized64.model', u'D:/Documents/Stage/python/models/test2/test2optimized65.model', u'D:/Documents/Stage/python/models/test2/test2optimized66.model', u'D:/Documents/Stage/python/models/test2/test2optimized67.model', u'D:/Documents/Stage/python/models/test2/test2optimized68.model', u'D:/Documents/Stage/python/models/test2/test2optimized69.model', u'D:/Documents/Stage/python/models/test2/test2optimized70.model', u'D:/Documents/Stage/python/models/test2/test2optimized71.model', u'D:/Documents/Stage/python/models/test2/test2optimized72.model', u'D:/Documents/Stage/python/models/test2/test2optimized73.model', u'D:/Documents/Stage/python/models/test2/test2optimized74.model', u'D:/Documents/Stage/python/models/test2/test2optimized75.model', u'D:/Documents/Stage/python/models/test2/test2optimized76.model', u'D:/Documents/Stage/python/models/test2/test2optimized77.model', u'D:/Documents/Stage/python/models/test2/test2optimized79.model', u'D:/Documents/Stage/python/models/test2/test2optimized80.model', u'D:/Documents/Stage/python/models/test2/test2optimized81.model', u'D:/Documents/Stage/python/models/test2/test2optimized82.model', u'D:/Documents/Stage/python/models/test2/test2optimized83.model', u'D:/Documents/Stage/python/models/test2/test2optimized84.model', u'D:/Documents/Stage/python/models/test2/test2optimized85.model', u'D:/Documents/Stage/python/models/test2/test2optimized86.model', u'D:/Documents/Stage/python/models/test2/test2optimized88.model', u'D:/Documents/Stage/python/models/test2/test2optimized89.model', u'D:/Documents/Stage/python/models/test2/test2optimized90.model', u'D:/Documents/Stage/python/models/test2/test2optimized91.model', u'D:/Documents/Stage/python/models/test2/test2optimized92.model', u'D:/Documents/Stage/python/models/test2/test2optimized93.model', u'D:/Documents/Stage/python/models/test2/test2optimized96.model', u'D:/Documents/Stage/python/models/test2/test2optimized97.model', u'D:/Documents/Stage/python/models/test2/test2optimized99.model', u'D:/Documents/Stage/python/models/test2/test2optimized100.model']
    e.openFiles(rest)
    e.removeTheWeak(1000)# a high number to remove the bogus models

##    e.showTimeGrid()
##    e.showTimeGrid(setNormalizeTo=False)

##    e.showHierarchy()
    e.truthFinder()
##    e.showPCA()
##    e.showHistory()

##    e.showVariables()
##    e.showFinalValuesGrid()
##    print('next!')
##    e.evaluateSolutionStability()

##    e.showFinalValuesGrid()
##    e.cluster()
##    e.parameterSensitivityAnalysis(e.goalGene)

