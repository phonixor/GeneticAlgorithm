import math
import numpy
import scipy
from scipy.integrate import odeint
import GA_gene
import random

class gene(GA_gene.GA_gene):
    """
        dx/dt= prodRate*sigma(minProd + x*k1 + y*k2 ...) - decay*x
        y being a second gene concentration
        k's being the k values :)

        sigma=@(x)(0.5*((x./sqrt((x.^2)+1))+1)); %matlab code
        it is a curve from 0 to 1, to simulate gene regulation

        t=numpy.linspace(0,self.endTime,101)<-- timepoints are currently hardcoded


        fitness requires absolute truth to be filled with the result of a gene (created by solveODE)

        todo: handle NaN fitness results... maybe randomize?
        todo: normalize divide by 0 error ~~ kinda rare

        todo: consider saving fitness value in models... to avoid load times.... may cause bugs though... might need a get results???

    """

    # Model constants parameters and stuff
    conservedNotes=''

    initialValues=[]
    simpleReactions=scipy.array([]) # contains an array of all the rate constants
    prodRate=[] # contains the production weight... and thus maximu
    minProd=[] # the default production without regulation, this is independent of any concentrations, and is inside the sigma
    decay=[] # the decay rate
    geneNames=[]
    variableNames=[] # TODO should contain the variable names used for displaying puposes and stuff ...
    # currently assumes files loaded are in the same order as the variable list generated here..


    # Model constraints
    minDecay=0
    maxDecay=0.99 # if 1 it may cause trouble? cause total of 0?
    minMinProd=-10
    maxMinProd=10
    minProdRate=0.01 # if 0 it may cause trouble cause a total of 0?
    maxProdRate=10
    minSimpleReaction=-10
    maxSimpleReaction=10
    # time for the ODE
    endTime=25
    # Model constraints in a single list
    minParRangeList=[]
    maxParRangeList=[]

    # the final objective, needed to calculate fitness
    absoluteTruth=[]

    # half of the model is a mirror to the other half, so don't be silly and calculate all!
    mirror=True # True is only calc half, false is calc all

    # storage for the results of the ODE
    results=None # data cube [cell, time, gene]
    normalized=True # normalize the results or not... fitness is calculated over the normalized results
    useTimeSeries=False # default compares only the last time point, if set to True it will evaluate over 101 time points




    # mutation mode
    mutationType=2  # 1 is a single mutation, over the full variable range
                    # 2 is mutate all a little (normal distribution centered around current value)

    # mutationType2 options
    # see math.ods for more info
    mutationType2STDEVModifier=0.025 # this param affect the STDEV for the
    mutationType2STDEVList=[]

    def __init__(self):
        pass

    def create(self):
        """overwritten method"""
        self.randomize()
        return self

    def randomize(self):
        """ randomize the individuel used for creating new ones form existing ones like with immigrants """
        # randomize decay
        for i in range(len(self.decay)):
            self.decay[i]=random.uniform(self.minDecay,self.maxDecay)

        # randomize minProd
        for i in range(len(self.minProd)):
            self.minProd[i]=random.uniform(self.minMinProd,self.maxMinProd)

        # randomize prodRate
        for i in range(len(self.prodRate)):
            self.prodRate[i]=random.uniform(self.minProdRate,self.maxProdRate)

        # randomize simple reactions
        # (maybe give 0 a bias)
        for i in range(self.simpleReactions.shape[0]):
            for j in range(self.simpleReactions.shape[1]):
                self.simpleReactions[i,j]=random.uniform(self.minSimpleReaction,self.maxSimpleReaction)

        # calculate fitness
        self.fitnessValue=None
        self.fitness()

    def getVariableList(self):
        """ get a list containing all the model variables/parameters, put them back into place with setVariableList"""
        variableList=[]
        variableList.extend(self.decay)
        variableList.extend(self.minProd)
        variableList.extend(self.prodRate)
        for i in range(self.simpleReactions.shape[0]):
            variableList.extend(self.simpleReactions[i,:])
        return variableList

    def setVariableList(self, variableList):
        """ put the variable list back into place
            assumes that the arrays already have the proper length/shape
        """
        index=0
        self.decay=variableList[index:index+len(self.decay)]
        index+=len(self.decay)
        self.minProd=variableList[index:index+len(self.minProd)]
        index+=len(self.minProd)
        self.prodRate=variableList[index:index+len(self.prodRate)]
        index+=len(self.prodRate)
        for i in range(self.simpleReactions.shape[0]):
            self.simpleReactions[i,:]=variableList[index:index+self.simpleReactions.shape[0]]
            index+=self.simpleReactions.shape[0]

    def createMaxMinList(self):
        """create 2 arrasy min and max values for each parameter/variable
           also creates variable names and stuff

        """
        self.minParRangeList=[]
        self.maxParRangeList=[]
        self.mutationType2STDEVList=[]
        self.variableNames=[]


        for i in range(len(self.decay)):
            self.minParRangeList.append(self.minDecay)
            self.maxParRangeList.append(self.maxDecay)
            self.mutationType2STDEVList.append((self.maxDecay-self.minDecay)*self.mutationType2STDEVModifier)
            self.variableNames.append('decay_'+str(i+1))
        for i in range(len(self.minProd)):
            self.minParRangeList.append(self.minMinProd)
            self.maxParRangeList.append(self.maxMinProd)
            self.mutationType2STDEVList.append((self.maxMinProd-self.minMinProd)*self.mutationType2STDEVModifier)
            self.variableNames.append('minProd_'+str(i+1))
        for i in range(len(self.prodRate)):
            self.minParRangeList.append(self.minProdRate)
            self.maxParRangeList.append(self.maxProdRate)
            self.mutationType2STDEVList.append((self.maxProdRate-self.minProdRate)*self.mutationType2STDEVModifier)
            self.variableNames.append('prodRate_'+str(i+1))
        for i in range(self.simpleReactions.shape[0]*self.simpleReactions.shape[1]):
            self.minParRangeList.append(self.minSimpleReaction)
            self.maxParRangeList.append(self.maxSimpleReaction)
            self.mutationType2STDEVList.append((self.maxSimpleReaction-self.minSimpleReaction)*self.mutationType2STDEVModifier)

        # fill: variableNames
        for i in range(1,self.simpleReactions.shape[0]+1):
            for j in range(1,self.simpleReactions.shape[1]+1):
                self.variableNames.append('gene_'+str(i)+'_'+str(j))
        # array love
        self.variableNames=scipy.array(self.variableNames)


    def mutate(self):
##        # create indexes
##        endDecay=len(self.decay)-1
##        endProdRate=endDecay+len(self.prodRate)
##        endMinProd=endProdRate+len(self.minProd)
##        totalVar=endMinProd+(self.simpleReactions.shape[0]*self.simpleReactions.shape[1])
##
##        selector=random.randint(0,totalVar)
##
##        if(selector<=endDecay):
##            # mutate decay
##            self.decay[selector]=random.uniform(self.minDecay,self.maxDecay)
##        elif(selector<=endProdRate):
##            # mutate prodrate
##            selector=(selector-endDecay)-1
##            self.prodRate[selector]=random.uniform(self.minProdRate,self.maxProdRate)
##        elif(selector<=endMinProd):
##            # mutate minProd
####            print(selector)
####            print(endProdRate)
####            print(self.minProd)
##            selector=(selector-endProdRate)-1
##            self.minProd[selector]=random.uniform(self.minMinProd,self.maxMinProd)
##        else:
##            selector=(selector-endMinProd)-1
##            x=int(selector/self.simpleReactions.shape[1])
##            y=selector-(self.simpleReactions.shape[0]*x)
####            print('simpleReactions')
####            print(x)
####            print(y)
##            self.simpleReactions[x,y]=random.uniform(self.minSimpleReaction,self.maxSimpleReaction)
##            # maybe get a bias for 0 here...
        # single mutation
        if self.mutationType==1:
            params=self.getVariableList() # get all the variables in a single list
            index=random.randint(0,len(params)-1)# select variable for mutation
            params[index]=random.uniform(self.minParRangeList[index],self.maxParRangeList[index])# mutate
            self.setVariableList(params) # put the variables back into place

        # global shift
        elif self.mutationType==2:
            params=self.getVariableList() # get all the variables in a single list
            for i in range(len(params)):
                # random shift normal distributed...
                params[i]=random.normalvariate(params[i],self.mutationType2STDEVList[i]) # (mean, stdev)
                # make sure it stays withing model constraints/bounds
                if params[i] < self.minParRangeList[i]:
                    params[i]=self.minParRangeList[i]
                if params[i] > self.maxParRangeList[i]:
                    params[i]=self.maxParRangeList[i]
            self.setVariableList(params) # put the variables back into place

        # should never get to the next part!
        else:
            print('ERROR!! mutation type not implemented')

        # recalculate fitness
        self.fitnessValue=None
        self.fitness()


    def crossover(self, other):
        """ not a proper crossover...
            but it are not real genes either
        """

##        # randomize decay
##        for i in range(len(self.decay)):
##            if random.random()>0.5:
##                switch=self.decay[i]
##                self.decay[i]=other.decay[i]
##                other.decay[i]=switch
##
##        # randomize minProd
##        for i in range(len(self.minProd)):
##            if random.random()>0.5:
##                switch=self.minProd[i]
##                self.minProd[i]=other.minProd[i]
##                other.minProd[i]=switch
##
##        # randomize prodRate
##        for i in range(len(self.prodRate)):
##            if random.random()>0.5:
##                switch=self.prodRate[i]
##                self.prodRate[i]=other.prodRate[i]
##                other.prodRate[i]=switch
##
##        # randomize simple reactions
##        # (maybe give 0 a bias)
##        for i in range(self.simpleReactions.shape[0]):
##            for j in range(self.simpleReactions.shape[1]):
##                if random.random()>0.5:
##                    switch=self.simpleReactions[i,j]
##                    self.simpleReactions[i,j]=other.simpleReactions[i,j]
##                    other.simpleReactions[i,j]=switch
        # get all the variables in a single list
        paramsSelf=self.getVariableList()
        paramsOther=other.getVariableList()

        # switch / recombinate / crossover
        for i in range(len(paramsSelf)):
            if random.random()>0.5:
                switch=paramsSelf[i]
                paramsSelf[i]=paramsOther[i]
                paramsOther[i]=paramsSelf[i]

        # put the variables back into place
        self.setVariableList(paramsSelf)
        other.setVariableList(paramsOther)

        # calculate fitness
        self.fitnessValue=None
        other.fitnessValue=None
        self.fitness()
        other.fitness()



    def load(self, filename):
        """ use this method to load a model file...

        """
##        filename='D:\\Documents\\Stage\\python\\models\\3geneWithVarProduction.model'

        f = open(filename, 'r')

##        print(f)

        line=f.readline()

        while(line != ''): # if its an empty row an end of line char would be there
            line=line.strip() #remove end of line / new line chars
            #
            #
            #CONSERVED_NOTES
            if(line=='#CONSERVED_NOTES'):
##                print('Reading conserved notes...')
                self.conservedNotes=''
                while(line != '#END'):
                    line=f.readline().strip()
                    if(line != '#END'):
                        self.conservedNotes=self.conservedNotes+line+"\n"
            #HISTORY
            elif(line == '#HISTORY'):
                # note that you lose some history by importing... (ID's are lost)
##                print('Parsing history...')
                self.history=[]
                while(line != '#END'):
                    line=f.readline().strip()
                    if(line != '#END'):
##                        self.history.append([line.split("\t")[0],float(line.split('\t')[1])])
                        self.history.append(line.split("\t"))

            #PER_GENE_OPTIONS
            elif(line == '#PER_GENE_OPTIONS'):
##                print('Parsing per gene options...')
                self.prodRate=[]
                self.minProd=[]
                self.decay=[]
                self.geneNames=[]
                perGeneOptionsHeader=None
                perGeneOptionsData=[]

                #first the header
                perGeneOptionsHeader=f.readline().strip().split("\t")

                #get the data
                while(line != '#END'):
                    line=f.readline().strip()
                    if(line != '#END'):
                        perGeneOptionsData.append(line.split('\t'))
                #convert it into an array for easy handling
                perGeneOptionsData=scipy.array(perGeneOptionsData)

                # identify the data
                counter=0
                for head in perGeneOptionsHeader:
                    if head == 'prodRate':
                        self.prodRate=perGeneOptionsData[:,counter].astype('float')
##                        for i in range(1,len(self.prodRate)+1):
##                            self.variableNames.append('prodRate_'+str(i))
                    elif head == 'minprod':
                        self.minProd=perGeneOptionsData[:,counter].astype('float')
##                        for i in range(1,len(self.minProd)+1):
##                            self.variableNames.append('minProd_'+str(i))
                    elif head == 'decay':
                        self.decay=perGeneOptionsData[:,counter].astype('float')
##                        for i in range(1,len(self.decay)+1):
##                            self.variableNames.append('decay_'+str(i))
                    elif head == 'name':
                        self.geneNames=perGeneOptionsData[:,counter]
##                        for i in range(len(self.geneNames)):
##                            self.variableNames.append(self.geneNames[i]) # silly me... this contains no variables...
                    counter+=1

            #GENE_V_DATA_RELATIONS
            elif(line == '#GENE_V_DATA_RELATIONS'):
##                print('Ignoring gene v data relations...')
                pass

            #SIMPLE_REACTIONS
            elif(line == '#SIMPLE_REACTIONS'):
##                print('parsing simple reactions...')
                self.simpleReactions=[]
                f.readline() # ignore header
                while(line != '#END'):
                    line=f.readline().strip()
                    if(line != '#END'):
                        self.simpleReactions.append(line.split('\t'))
                # convert it back into an array
                self.simpleReactions=scipy.array(self.simpleReactions)
                self.simpleReactions=self.simpleReactions[:,:-1].astype('float') # remove names at the end and cast it to float
##                print(self.simpleReactions)

##                # fill: variableNames
##                for i in range(1,self.simpleReactions.shape[0]+1):
##                    for j in range(1,self.simpleReactions.shape[1]+1):
##                        self.variableNames.append('gene_'+str(i)+'_'+str(j))

            #COMPLEX_REACTIONS
            elif(line == '#COMPLEX_REACTIONS'):
##                print('Ignoring complex reactions...')
                pass

            #SETTINGS
            elif(line == '#SETTINGS'):
##                print('Ignoring settings...')
                pass


            #INITIAL_VALUES
            elif(line == '#INITIAL_VALUES'):
##                print('Parsing initial values...')
                self.initialValues=[]
                while(line != '#END'):
                    line=f.readline().strip()
                    if(line != '#END'):
                        self.initialValues.append(line.split("\t"))
                self.initialValues=scipy.array(self.initialValues)
                self.initialValues=self.initialValues[:,:-1].astype('float')

            #Anything else
            else:
                #Probably nothing of interest...
                pass

            # get the next line
            line=f.readline()

        f.close() # release the hostage
##        self.variableNames=scipy.array(self.variableNames) # arrays are lovely for indexing and stuff that should be in standard..
        self.createMaxMinList()


    def save(self, filename):
        """ use this model to store a model file
        """
        # file
        f = open(filename, 'w')
        s=''
        # conserved notes
        s+='#CONSERVED_NOTES\n'
        s+=self.conservedNotes
        s+='#END\n'
        s+='\n'
        # history
        s+='#HISTORY\n'
        for generation in self.history:
            for stuff in generation:
                s+=str(stuff)+'\t'
            s=s[:-1]+'\n' # remove extra tab and add end of line char
        s+='#END\n'
        s+='\n'
        # per gene options
        s+='#PER_GENE_OPTIONS\n'
        s+='prodRate\tminprod\tdecay\tname\n'
        for i in range(len(self.geneNames)):
            s+=str(self.prodRate[i])+'\t'+str(self.minProd[i])+'\t'+str(self.decay[i])+'\t'+self.geneNames[i]+'\n'
        s+='#END\n'
        s+='\n'
        # gen v data relations
        s+='#GENE_V_DATA_RELATIONS\n'
        for i in range(len(self.geneNames)):
            s+=self.geneNames[i]+'\n'
        s+='#END\n'
        s+='\n'
        # simple reactions
        s+='#SIMPLE_REACTIONS\n'
        for i in range(len(self.geneNames)):
            s+=self.geneNames[i]+'\t'
        s=s[:-1]+'\n'# remove extra tab and add end of line char
        for i in range(len(self.geneNames)):
            for j in range(len(self.geneNames)):
                s+=str(self.simpleReactions[i,j])+'\t'
            s+=self.geneNames[i]+'\n'
        s+='#END\n'
        s+='\n'
        # complex reactions... dont have those
        s+='#COMPLEX_REACTIONS\n'
        s+='#END\n'
        s+='#COMPLEX_REACTIONS_VALUES\n'
        s+='#END\n'
        s+='\n'
        # Settings
        # important if you want to add them in matlab... though that is bugged..
        s+='#SETTINGS\n'
        s+='sigma=@(x)(0.5*((x./sqrt((x.^2)+1))+1));\n'
        s+='%matlab code here....\n'
        s+='this.minDecayParameterRange=0;\n'
        s+='this.maxDecayParameterRange=1;\n'
        s+='this.minProdParameterRange=-10;\n'
        s+='this.maxProdParameterRange=10;\n'
        s+='this.freeSimpleReactions=1; %maybe this should be in the settings, and not in the actual model...\n'
        s+='#END\n'
        s+='\n'
        s+='#INITIAL_VALUES\n'
        for i in range(self.initialValues.shape[0]):
            for j in range(self.initialValues.shape[1]):
                s+=str(self.initialValues[i,j])+'\t'
            s+=self.geneNames[i]+'\n'
        s+='#END\n'


        f.write(s)
        f.close()


    def fitness(self):
        """ calculate the fitness value of this individual

            normalized - normalize the results and judge over the normalized results
            useTimeSeries - default compares only the last time point, if set to True it will evaluate over 101 time points
        """

        # check if its not already calculated...
        if(self.fitnessValue!=None):
            return self.fitnessValue

        # check if the fitness condition is present
        if(self.absoluteTruth==[]):
            print('ERROR! the fitness condidtion is not present!')
            raise Exception('DIE!!!  ERROR! the fitness condidtion is not present!')

        # do the acutal love
        self.results=self.solveODE()


        if self.useTimeSeries:
            difference=self.absoluteTruth-self.results
            length=difference.shape[0]*difference.shape[1]*difference.shape[2]
            difference=difference**2
            self.fitnessValue=math.sqrt(difference.sum()/length)
        else:
            difference=self.absoluteTruth[:,-1,:]-self.results[:,-1,:] # difference of last timepoint only
##            length=difference.shape[0]*difference.shape[1] # this is what it was... it is wrong... i took time... it should have been gene
##            # no apprently that gives an error
##            print(difference.shape)
##            length=difference.shape[0]*difference.shape[2]
            length=difference.shape[0]*difference.shape[1]
            difference=difference**2
            self.fitnessValue=math.sqrt(difference.sum()/length)
##        print(self.fitnessValue)

##        print(difference.shape)
##        print(difference.sum(axis=0).shape)
##        print(difference**2)


##	    self.initialValues=math.sqrt(sum(n*n for n in num)/len(num))
        #sqrt(sum(n*n for n in num)/len(num))

        if math.isnan(self.fitnessValue):
            print('WARNING!!!: No proper fitnessValue was calculated... returned a fitnessValue of 10000000')
            self.fitnessValue=10000000 # this should be large enough...

        return self.fitnessValue

    def sigma(self, value):
        """ function that returns a value between 0-1 with a slope around when value=0
            see the model for more information

        """
        return 0.5*((value/math.sqrt((value**2)+1))+1)

    def solveODE(self):
        """ does what is says... """
        t=numpy.linspace(0,self.endTime,101) # timepoints returned,  101 is the nr of time points
##        print(t)
        y0=numpy.array([1,2,3]) # initial values for t=0

##        self.simpleReactions=scipy.array([1,1])
##        self.simpleReactions=scipy.arange(12)

        nrOfCells=self.initialValues.shape[1]
        self.results=scipy.zeros([nrOfCells,t.shape[0],self.initialValues.shape[0]]) # data cube [cell, time, gene]

        if self.mirror: # make sure to only calc half
            nrOfCells=nrOfCells/2 # cause lying is cheap!

        # odeint returns a 2D array (time,gene)
        for i in range(nrOfCells):
            self.results[i,:,:]=odeint(self.f, self.initialValues[:,i], t,args=())
##        self.results[50,:,:]=odeint(self.f, self.initialValues[:,50], t,args=())# just 1 for testing purposes

        if self.normalized:
        # normalize
            # devide all values by the maximum value of that gene
            # TODO what if 0? gota a: RuntimeWarning: invalid value encountered in divide
            self.results=self.results/self.results.max(axis=0).max(axis=0)

        if self.mirror:
            self.results[nrOfCells:,:,:]=self.results[nrOfCells-1::-1,:,:] # copy mirror paste

        return self.results

    def f(self,y,t=0):
        """ the model!
            this was the matlab code, so it has to do this
            dgene1dt=k10*sigma(0+k1*gene1+k2*gene2+k3*gene3+k13)-k16*gene1;
        """
        # inside the sigma part:
        # - the always on production
        dy=scipy.array(self.minProd)
        # - the simple reaction
        for i in range(self.simpleReactions.shape[0]):
            for j in range(self.simpleReactions.shape[0]):
                dy[i]=dy[i]+(y[j]*self.simpleReactions[i,j])

        # apply the sigma part
        for i in range(len(dy)):
            # weight*sigma
            dy[i]=self.prodRate[i]*self.sigma(dy[i])

        # apply the decay
        for i in range(len(dy)):
            dy[i]=dy[i]-self.decay[i]*y[i]

        return dy
##        function dy = tempmodel(t, y, rateConstants)
##        gene1 = y(1);
##        gene2 = y(2);
##        gene3 = y(3);
##        k1 = rateConstants(1);
##        k2 = rateConstants(2);
##        k3 = rateConstants(3);
##        k4 = rateConstants(4);
##        k5 = rateConstants(5);
##        k6 = rateConstants(6);
##        k7 = rateConstants(7);
##        k8 = rateConstants(8);
##        k9 = rateConstants(9);
##        k10 = rateConstants(10);
##        k11 = rateConstants(11);
##        k12 = rateConstants(12);
##        k13 = rateConstants(13);
##        k14 = rateConstants(14);
##        k15 = rateConstants(15);
##        k16 = rateConstants(16);
##        k17 = rateConstants(17);
##        k18 = rateConstants(18);
##        % *** Data to Gene Conversions ***
##        % *** Differential Equations ***
##        sigma=@(x)(0.5*((x./sqrt((x.^2)+1))+1));;
##        dgene1dt=k10*sigma(0+k1*gene1+k2*gene2+k3*gene3+k13)-k16*gene1;
##        dgene2dt=k11*sigma(0+k4*gene1+k5*gene2+k6*gene3+k14)-k17*gene2;
##        dgene3dt=k12*sigma(0+k7*gene1+k8*gene2+k9*gene3+k15)-k18*gene3;
##        % *** Gene to Data Conversions ***
##        % *** Return the dt's of the data ***
##        dy = [dgene1dt; dgene2dt; dgene3dt];
##        end

    def displayResults(self):
        # data cube [cell, time, gene]
        print("Plotting the sexy stuff...")
        import pylab # put on this level as lisa has no pylab
##        pylab.plot(self.results[50,:,:])
##        pylab.show()
        print(self.results.shape)
        for i in range(self.results.shape[2]):
            pylab.plot(self.results[:,-1,i],label=self.geneNames[i])
        pylab.xticks([50],[""])
        pylab.xlabel('Cells')
        pylab.ylabel('Normalized Concentration')
        pylab.text(1,1.03,'time=25') #might need something not hardcoded.... like time itself :P
        pylab.legend(loc='upper left')
        pylab.show()

##        test=[[1,2],[2,4],[4,5],[5,5]]
##        time=[1,2,3,4]
##        pylab.plot(time,test)
##        pylab.show()

    def displayTimeSeries(self):
        """ display the result over time, 4 images"""
##        nrOfPlots=4
        import pylab
        timepoints=numpy.linspace(0,self.endTime,5) #last time point will not be displayed because its nicer as a big picture...
        print(timepoints)
        pylab.figure()
        print(range(0,5))
        for i in range(0,4):
            pylab.subplot(1,4,i+1)
            pylab.plot(self.results[:,i*25,:])
            pylab.xticks([50],[""])
            pylab.title('time: '+str(timepoints[i]))


        pylab.show()

    def displayResults2(self):
        """ show results with time slider

            geneNames have to be unique!

        """
##
##        import pylab
####        from matplotlib.widgets import Slider
##        mi=1
##        ma=self.results.shape[1]
##        print(ma)# should be 101
##
##        figure=pylab.figure()
##
##        ax=pylab.Axes(figure,[0.15, 0.1, 0.65, 0.03])
##        slider=pylab.Slider(ax=ax,label='time slider',valmin=mi,valmax=ma,valinit=1)
##
##        def update():
##            pylab.plot(self.results[:,slider.val,:])
##            pylab.draw()
##
##        slider.on_changed(update)
##        pylab.show()

        import pylab
        import scipy
        from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons


##        test = pylab.plot(self.results[:,-1,:])
##        pylab.show()

        # results is a data cube [cell, time, gene]
        miTime=0
        maTime=self.results.shape[1]-1
##        xdata=range(self.results.shape[0])

        ##ax = pylab.subplot(111)

        pylab.subplots_adjust(left=0.25, bottom=0.25)
##        t = scipy.arange(0.0, 1.0, 0.001)
##        a0 = 5
##        f0 = 3
##        s = a0*scipy.sin(2*scipy.pi*f0*t)
##        plot, = pylab.plot(t,s, lw=2, color='red')
        selection=scipy.ones(len(self.geneNames)).astype(bool)# select all on start
        plots = pylab.plot(self.results[:,maTime,:]) # apperently returns a plot for each line...


##        pylab.axis([0, 1, -10, 10])


        axcolor = 'lightgoldenrodyellow'
##        axfreq = pylab.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
##        axamp  = pylab.axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)

##        sfreq = Slider(axfreq, 'Freq', 0.1, 30.0, valinit=f0)
##        samp = Slider(axamp, 'Amp', 0.1, 10.0, valinit=a0)
        ax=pylab.axes([0.15, 0.1, 0.65, 0.03])
        slider=pylab.Slider(ax=ax,label='time slider',valmin=miTime,valmax=maTime,valinit=maTime)

        def update(val):
##            amp = samp.val
##            freq = sfreq.val
##            l.set_ydata(amp*scipy.sin(2*scipy.pi*freq*t))
            for i in range(len(plots)):
                plots[i].set_ydata(self.results[:,slider.val,i])
                plots[i].set_visible(selection[i])

            pylab.draw()

##        sfreq.on_changed(update)
##        samp.on_changed(update)
        slider.on_changed(update)

        resetax = pylab.axes([0.8, 0.025, 0.1, 0.04])
        button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
        def reset(event):
##            sfreq.reset()
##            samp.reset()
            slider.reset()
        button.on_clicked(reset)


        rax = pylab.axes([0.025, 0.5, 0.15, 0.15], axisbg=axcolor)
        checker=CheckButtons(rax,self.geneNames,actives=selection)
        def selector(val):
##            print(val)
##            print(scipy.array(range(len(self.geneNames)))[self.geneNames==val][0])
            geneNr=scipy.array(range(len(self.geneNames)))[self.geneNames==val][0] # its retarded to check label names... but that is the way they like it....
            selection[geneNr]=not(selection[geneNr])
            update(slider.val)
        checker.on_clicked(selector)
##        print(checker.eventson)
##        print(checker.drawon)

##        radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)


##
##        rax = pylab.axes([0.025, 0.5, 0.15, 0.15], axisbg=axcolor)
##        radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)
##        def colorfunc(label):
##            for i in range(len(plots)):
##                plots[i].set_color(label)
####            plots.set_color(label)
##            pylab.draw()
##        radio.on_clicked(colorfunc)

        pylab.show()




    def localOptimize(self):
        """ use the scipy function fmin_slsqp to optimize to do a local search
            this was the only local optimize method that accepted bounds...

        """
        # if a GA just isnt good engough :)
        print('starting a local optimalization\search...')
        x0=self.getVariableList()
##        bounds=[self.minParRangeList,self.maxParRangeList]
        bounds=[]
        for i in range(len(self.minParRangeList)):
            bounds.append([self.minParRangeList[i],self.maxParRangeList[i]])
##        print(bounds)
        # store old fitness
        oldFitnessValue=self.fitness()
##        # http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
##        from scipy.optimize import ??? to fancy!
##        # scipy.optimize.minimize(fun, x0, args=(), method='BFGS', jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)[source]
##        # Bounds for variables (only for L-BFGS-B, TNC, COBYLA and SLSQP). (min, max) pairs for each element in x, defining the bounds on that parameter.
##        # Use None for one of min or max when there is no bound in that direction.
##
##        results=minimize(self.f,x0,bounds=bounds)

        # need scipy > 0.11 for orther local optimizers
        # https://www.surfsara.nl/nl/systems/lisa/news/python-2.7.2
        # whcih is not available there
##        print('before..')
##        print(self.fitness())
##        print(x0)
##        print('-----------------------')

        from scipy.optimize import fmin_slsqp
        result = fmin_slsqp(self.fitness_ForLocalOptimize, x0, bounds=bounds) # returns the optimized variable list

##        from scipy.optimize import fsolve
##        results=fsolve(self.fitness_ForLocalOptimize,x0,band=bounds)

##        print('-----------------------')
##        print('after...')
##        print(result)
##        print(self.fitness())
        self.setVariableList(result) # save the result

        self.history.append(['localOptimize', self.fitnessValue, oldFitnessValue])



    def fitness_ForLocalOptimize(self, newVariableList): # might wanna rename it :P
        """ the new variable list is not restored!!! so copy the model you want to optimize!

        """
        # set the variable list as the values
        self.setVariableList(newVariableList)
        self.fitnessValue=None # make sure to recalculate the fitness
        # run the fitness
        return self.fitness()










if __name__ == '__main__':
    test=gene()
##    print(test.sigma(100))
##    print(test.sigma(-100))

##    filename='D:/Documents/Stage/python/models/3geneWithVarProduction.model'
    filename='D:/Documents/Stage/python/models/4geneWithVarProduction.model'
    test.load(filename)
    test.solveODE()
##    test.displayTimeSeries()
##    test.displayResults()
    test.displayResults2()

##    filename='D:\\Documents\\Stage\\python\\models\\4geneWithVarProduction.model'
##    test.load(filename)
##    test.normalized=False
##    test.useTimeSeries=True
##    test.solveODE()

##    test.displayResults2()
##    filename2='D:\\Documents\\Stage\\python\\models\\testmodel.model'
##    test.save(filename2)
##
##    # test if it still works!!
##    test2=gene()
##    test2.load(filename2)
##    test2.solveODE()
##    test2.setVariableList(test2.getVariableList())#this should change nothing!
##    test2.displayResults()
##    test2.save(filename2)
##
##    test3=gene()
##    test3.load(filename2)
##    test3.solveODE()
##    test3.displayResults()
##
##    print('local optimize')
##    test3.absoluteTruth=test3.solveODE()
##    test3.randomize()
##    test3.localOptimize()
##    test3.displayResults()
##    test4=gene()
##    test4.load('D:\\Documents\\Stage\\python\\models\\test4\\optimized1-1.model')
##    test4.absoluteTruth=test.results
##    test4.normalized=False
##    test4.useTimeSeries=True
##    test4.fitness()
##    test4.displayResults2()




