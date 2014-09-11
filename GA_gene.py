class GA_gene:
    """ GA_gene genetic algoritm

        implementation of a genetic algorithm using GA_population with GA_gene to do magic!
        both need to be inherented so you can give your own implementations of mutation and crossover and fitness and stuff...


        the fitnessValue should be kept upToDate after any changes to the model!
        the lower its value the better it is!

        history contains:

            if:
                create - 'create', fitnessValue, geneID
                mutate - 'mutate', fitnessValue
                crossover - 'crossover', fitnessValue, geneID of crossover partner, fitnessValue of crossover partner


    """

    fitnessValue=None # the fitnessValue of the current gene (should be set to None if anything in the model changes...)
    ID=0 # ID number given on creation, so multiple copies can exicst if mutated or crossover copies are preserved
    __ID__=0 # class ID counter
    history=[]

    def __init__(self):
        pass

##    def load(self, filename):
##        """... load a individual from a file, this function needs to be overwritten"""
##        pass
    def __create__(self):
        # create unique identifier
        GA_gene.__ID__+=1
        self.ID=GA_gene.__ID__
        self.create()
        self.history=[['create', self.fitnessValue, self.ID]]
        return self

    def create(self):
        """ create a new individual, this function needs to be overwritten
            its fitness should be calculated using the implementation of fitness()
        """
        return self


    def fitness(self):
        """ calculate the fitness of this indivdual this needs to be overwritten should return the fitness value
            aswell as storing it in fitnessValue
        """
        pass

    def __mutate__(self):
        self.mutate()
        self.history.append(['mutate', self.fitnessValue])

    def mutate(self):
        """ mutate this individual, this function needs to be overwritten
            its fitness should be calculated using the implementation of fitness()
        """
        pass

    def __crossover__(self, other):
        self.crossover(other)
        self.history.append(['crossover', self.fitnessValue, other.ID, other.fitnessValue])
        other.history.append(['crossover', other.fitnessValue, self.ID, self.fitnessValue])

    def crossover(self, other):
        """ mutate this individual, this function needs to be overwritten
            its fitness should be calculated using the implementation of fitness()
        """
        pass


if __name__ == '__main__':
    pass