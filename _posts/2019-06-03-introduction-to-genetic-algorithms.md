---
author: Philip Kiely
date: 2019-06-03 14:23:43 +0000
excerpt: Genetic algorithms are a specific approach to optimization problems that
  can estimate known solutions and simulate evolutionary behavior in complex systems.
feature_image: /assets/images/hero/introduction-to-genetic-algorithms-hero.jpg
layout: post
slug: introduction-to-genetic-algorithms
tags: '[]'
title: Introduction to Genetic Algorithms
---

When you're solving a problem, how do you know if the answer you've found is correct? In many domains, there is a single correct answer. A mathematical function may have a global maximum or other well-defined attributes. However, other problems, like how a cell behaves in a petri dish, do not have clear solutions. Enter evolution, which does not design towards a known solution but optimizes around constraints. Genetic algorithms are a specific approach to optimization problems that can estimate known solutions and simulate evolutionary behavior in complex systems.

This article will briefly discuss the terms and concepts required to understand genetic algorithms then provide two examples. The first example will be estimating the optimal inputs and maximum value of a multivariable function. The second example will develop a simple simulation of cooperating and non-cooperating species in a resource-constrained environment to explore how genetic algorithms can model complex behavior. I will leave you with a template for writing your own genetic algorithms and ideas for domains and problems to address with the technique. Genetic algorithms are a broad, complex, and actively researched subject. This article serves as an introduction to the concepts and techniques.

### Ready to build, train, and deploy AI?

#### Get started with FloydHub's collaborative AI platform for free

###### [Try FloydHub for free ](https://www.floydhub.com/?utm_source=blog&utm_medium=banner-intro-ga&utm_campaign=try_floydhub_for_free)

## Structure of a Genetic algorithm

![](https://lh6.googleusercontent.com/na4CKBbtZukfpjhS-ThB-YgAMqoPRMQASTDSo6norGJnWmHRMFMgPGuoc84xeFoPmhXWhd7KnsGyfFD8Bs7A5VYP5IiqGHakE7wJ3cr0efQhhNBVvue8YFM3OZMZVEAlOiIOt_TP)A flow chart describing the basic structure of a genetic algorithm

Genetic algorithms vary in their structure based on their purpose, but all of them share a few common components. The algorithm begins by initializing **a population of individuals** using default or random values. Then, it runs each member of that population through a fitness function. It selects the fittest members of the population to reproduce using a method defined in the **reproduction function** , then repeats the evaluation and reproduction until a desired number of iterations have passed. At termination, the algorithm presents the best member or members of the population according to the fitness function. Let's discuss each of these concepts further.

Each individual in the population must be represented by a value or object, which must be stored in a data structure. A basic example is each individual being a tuple of values and the population is a list of such tuples. An individual can also be an object or class, with whatever attributes and methods you implement. The population data structure may also place constraints upon the simulation, for example, the distance between two individuals in a list or two-dimensional array may be important in some functions.

![](https://lh4.googleusercontent.com/cNX1shj2-nPdGlulva4UfUdNWwmWrgOSDviyX4tEIZU3Vufw60Qu--2_Yf1ecvaI_vKb0Ha3mRxUIMY5PpJMXpGTvIB0oR0QPu5yVJ69QhqgAjtf2BI_kWVAQB0-xAcd8JDpsybL)Source: [How to define a Fitness Function in a Genetic Algorithm?](https://towardsdatascience.com/how-to-define-a-fitness-function-in-a-genetic-algorithm-be572b9ea3b4)

### Fitness Function

The **fitness function** is the heart of a genetic algorithm. The function takes an individual and determines how well it fulfills whatever criteria the algorithm is optimizing for. If you were writing a genetic algorithm that simulated a frog jumping, the fitness function might be the height of the jump given weight, leg size, and energy constraints. The fitness function should be applied to each individual of the population separately to determine whether they should be allowed to reproduce. The function may return a fitness score or a boolean for whether the individual passes a set threshold for reproduction.

### Selection Function

The **selection function** takes the population and the results of the fitness function to determine who should reproduce. If the fitness function had a set threshold for reproduction and returned a boolean, then the selection function simply filters the population by that value. However, if the fitness function returned raw scores, the selection function calculates a threshold from those scores. For example, it may calculate the average score and only keep the top half of the population. It passes the selected population into the reproduction function and deletes the rejected individuals like Thanos snapping his fingers.

![](https://lh6.googleusercontent.com/V-PvjHSGZmKYCOJBnHe0AfQw8vdvDwPF30xxsLTR4kobH3nXo5IhxF-mtyqCLDTLr2MKu-M__FHiW8ci1aVeYmpfi1LUn7QS_wHiLqhJAnNuAhiaznjY2mldmif1FeP1GIv62pAT)Image from [Wikimedia Commons ](https://commons.wikimedia.org/wiki/File:Mutation_and_selection_diagram.svg)

### Reproduction Function

The **reproduction function** determines how to expand the population based on the existing members. Modifying the behavior and hyperparameters of the reproduction function is one of the most complex and impactful parts of creating a genetic algorithm, as the reproduction function is what determines how the population changes over time. The simplest reproduction method is **mutation** , where each new member is based on a single individual. If you are doubling the population by mutation, for each individual create a new individual with the same characteristics modified by a random factor. **Crossover** is a more complex method of reproduction, where each new individual is based on some combination of existing individuals. Crossover still mutates the new organism’s attributes, but does so by applying a function of multiple organisms’ attributes. These two approaches can simulate asexual and sexual reproduction, respectively, and both include random factors to advance the population as a whole and model the role of chance in real-world evolution. The reproduction function returns the new population, which may be the same size or a different size than the original population.

### **Termination** Function

After the desired iterations have occurred, the **termination function** takes the ending population and returns the best members by fitness score. The role of the termination function depends entirely on the application. For a video game, the termination function might output optimal statistics for the final boss, for a mathematical optimization, it returns the best input values to the function.

Now, we will consider these functions in action by implementing and evaluating two genetic algorithms.

## Example: Maximization

![](https://lh3.googleusercontent.com/QY-BRbZ-fmPFSluzvsJ9SA5BjL1wnW4Yo5GNaE-BAKtStU8uHlM6bH9PfEH-IEVK-VNxUap3F7o3X1XRcUMCzZ4yGNjscnpG08NToKmkdhZjYF0upp73xHXy4QhZMDLy3WnFpCzx)Graph generated using WolfRamAlpha

Any university student taking Calculus will tell you that solving for the maximum value of a function of multiple variables is a tedious operation. While we could use a solver to determine an exact solution to a given problem, we will instead use a genetic algorithm to find an approximate solution. While this isn't a great real-world example, most problems that are simple enough for a first example are simple enough that you can solve them without a genetic algorithm.

Consider the function: $$-x^2 + 2x -y^2 + 4y$$ If you have a background in calculus, you might be able to calculate the values of x and y that maximize the function in a couple of minutes. However, for more complicated functions like: $$-2w^2 + \sqrt{|w|} -x^2 + 6x - y^2 - 2y - z^2 + 4z$$ this becomes a lengthy exercise. We will implement a genetic algorithm that solves this problem.

[![Run on FloydHub](https://static.floydhub.com/button/button.svg)](https://floydhub.com/run?template=https://github.com/philipkiely/floydhub_genetic_algorithm_tutorial)
    
    
    # A system that uses a genetic algorithm to maximize a function of many variables
    import random
    import sys
    import math
    # Fitness_function of four variables with a maximum at (w=-.25, x=3, y=-1, z=2)
    def fitness_function(w, x, y, z):
        return -2 * (w ** 2) + math.sqrt(abs(w)) - (x ** 2) + (6 * x) - (y ** 2) - (2 * y) - (z ** 2) + (4 * z)
    # Simpler fitness_function of two variables with a maximum at (x=1, y=2)
    def simple_fitness_function(x, y):
        return - (x**2) + (2 * x) - (y ** 2) + (4 * y)
    

We begin by defining fitness functions that simply calculate the result of the functions we are trying to maximize. These fitness functions each take individual attributes and return a real number as a fitness score.
    
    
    # Takes a function and list of arguments, applies function to arguments
    def evaluate_generation(population):
        scores = []
        total = 0
        for individual in population:
            if len(individual) == 2:
                r = simple_fitness_function(individual[0], individual[1])
                scores.append(r)
                total += r
            elif len(individual) == 4:
                r = fitness_function(individual[0], individual[1], individual[2], individual[3])
                scores.append(r)
                total += r
            else:
                print("error: Wrong number of arguments received")
        avg = total / len(scores)
        return scores, avg
    

This function applies the fitness function to each member of the population and determines an average score. This average score will be used as a cut-off point for the selection function: only individuals with an above-average score (half of the population) will reproduce, the other half will be removed.
    
    
    # Create child from parent
    def mutate(individual):
        new = []
        for attribute in individual:
            new.append(attribute + random.normalvariate(0, attribute + .1))  # Random factor of normal distribution
        return new
    

The mutation function is a helper function for the reproduction function. It modifies each attribute of an individual by a random factor. In particular, we use a normalvariate distribution that has a higher chance of a smaller modification as it has a mean near zero. When creating new individuals, it is important to consider how different they should be from the parent(s). If the difference is too small, the algorithm will require many iterations to reach a max, but if the difference is too big, the output will be imprecise and may miss the maximum.
    
    
    # Given a population, return the best individual and the associated value
    def find_best(population):
        best = None
        val = None
        for individual in population:
            if len(individual) == 2:
                r = simple_fitness_function(individual[0], individual[1])
                try:
                    if r > val:
                        best = individual
                        val = r
                    except:  # On the first run, set the result as best
                        best = individual
                        val = r
            elif len(individual) == 4:
                r = fitness_function(individual[0], individual[1], individual[2], individual[3])
                try:
                    if r > val:
                        best = individual
                        val = r
                    except:  # On the first run, set the result as best
                        best = individual
                        val = r
            else:
                print("error: Wrong number of arguments received")
        return best, val
    

This function is the last step before output. After all of the iterations have run, this function takes the final population and evaluates it on the fitness function to return the best individual. In this case, the best individual is the estimated maximum value of the function and its attributes are the estimated optimal inputs to said function. The inputs may be greater or less than the actual best inputs, but the estimated maximum value will by definition be less than or equal to the true global maximum.
    
    
    # Create a population of p lists of [0, 0, ..., 0] of length n
    def initialize(n, p):
        pop = [[0] * n]
        for i in range(p):
            pop.append(mutate(pop[0]))
        return pop
    

Initialize simply creates a population of the desired size where each individual has the desired number of attributes. Each individual starts with every attribute value as zero.
    
    
    # Handle the output of the genetic algorithm
    def termination(best, val, total_iterations, population_size, num_attributes):
        best = [round(x, 3) for x in best]  #  Round for printing
        print("Ran", total_iterations, "iterations on a population of", population_size)
        print("The optimal input is", best, "with a value of", round(val, 3))
        if num_attributes == 2:
            print("The known maximum is at [1, 2] with a value of 5")
        elif num_attributes == 4:
            print("The known maximum is at [-.25, 3, -1, 2] with a value of 14.375")
        else:
            print("Error: Unsupported Individual Length")
    

In this case, the termination function simply prints the best result to the terminal. If the genetic algorithm was calculating values for some other function, the termination function would pass said values to that function.
    
    
    # Main function runs when the script is called
    if __name__ == "__main__":
        if len(sys.argv) != 4: # attrs switches between simple and full fitness_function, pop determines population size, iter determines iterations
            print("Usage: python geneticmax.py attrs(2 or 4) pop iter")
            exit()
        num_attributes = int(sys.argv[1])
        population_size = int(sys.argv[2])
        total_iterations = int(sys.argv[3])
        population = initialize(num_attributes, population_size)
        for iteration in range(total_iterations):
            scores, avg = evaluate_generation(population)
            deleted = 0
            new_population = []
            for i in range(len(population)):
                if scores[i] < avg:
                    deleted += 1
                else:
                    new_population.append(population[i])
            for i in range(deleted):
                new_population.append(mutate(new_population[i % len(new_population)])) # iterate over population with overflow protection
            population = new_population
        best, val = find_best(population)
        termination(best, val, total_iterations, population_size, num_attributes)
    

This main function manages the entire genetic algorithm. We take the number of attributes, population size, and number of iterations as command-line arguments. The number of attributes can be 2 or 4, which changes the fitness function that runs, while population size and total iterations can be any positive integer.  
Running `python geneticmax.py 2 100 100` gives us the following:
    
    
    Ran 100 iterations on a population of 100
    The optimal input is [1.004, 2.025] with a value of 4.999
    The known maximum is at [1, 2] with a value of 5
    

Running the simpler function on 100 individuals with 100 iterations got us within one one-thousandth of the best answer almost instantly. We can look at this estimate and, even if we don't know the maximum is at (x=1, y=2), by rounding we would understand that those would be values worth checking. Running the algorithm with more iterations or a larger population will usually, but not always, result in a higher value, and on this example even 10 iterations on a population of 100 should give a result over 4.9. See what you get when varying the population size and number of iterations.  
Let's try the same thing on the larger function. Try `python geneticmax.py 4 100 100`:
    
    
    Ran 100 iterations on a population of 100
    The optimal input is [-0.062, 2.682, -0.351, 2.232] with a value of 13.664
    The known maximum is at [-.25, 3, -1, 2] with a value of 14.375
    

Here, with four attributes, only 100 iterations leaves us well below the known maximum. Because there are more attributes, their combined error will be higher, which is why we need more trials to get a better result. Running this same input enough times will eventually get you a good result, but increasing the number of iterations is the best way to get a useful answer. Trials with 100,000 iterations on the same population size each took my computer about 20 seconds to run and yielded values above 14.365, with the highest at about 14.374. Again, this is close enough to give us a good idea of the actual best inputs and outputs.  
While linear programming or simple calculus might offer more straightforward methods of optimizing functions, these examples show that genetic algorithms can generate quick estimates for complicated functions.

## Example: Petri Dish

![](https://lh6.googleusercontent.com/PBF0E9xSsCN72JQy9iaspvWBx1XSu7meA8-mjAmBepGUksjq5zUIrBf0b7mXC5fuCMRzTbg-EyQqM33kHysw2pdBFFY3H1kbbbV_SkTrYlZWanYyQzPpk6SbmuUvGOK10kzM0FGW)Photo by[ Michael Schiffer](https://unsplash.com/photos/13UugSL9q7A?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on[ Unsplash](https://unsplash.com/search/photos/cell?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

Now that we've evaluated individuals using a genetic algorithm, let's write one to simulate population behavior.

You can write it using the code in this article or: [![Run on FloydHub](https://static.floydhub.com/button/button.svg)](https://floydhub.com/run?template=https://github.com/philipkiely/floydhub_genetic_algorithm_tutorial)

Imagine that there are a bunch of single-cell organisms in a petri dish. These organisms generate energy at a rate of one unit per round of time. Once they generate ten units of energy, they can use that energy to generate a new organism of the same type if there is sufficient space in the dish. If there are more organisms that can reproduce than space for new organisms, the organisms with the most energy will reproduce, so the fitness function is simply a read of an organism's energy. Every time it gains a unit of energy, the organisms also have a two percent chance of dying.  
There are two types of organisms in the dish: cooperators and non-cooperators. Each round, every cooperator will give away one unit of energy, which grants eight nearby organisms one unit of energy. Non-cooperators will not give away their energy but will accept energy from nearby cooperators. We run the simulation to determine the composition of the final population. These organisms are described the following object:
    
    
    # A simple organism
    class Organism():
    
        def __init__(self, cooperator):
            self.cooperator = cooperator
            self.energy = 1
            self.alive = True
    
        # if the organism is a cooperator, it will share energy
        def interact(self, population):
            if self.cooperator and self.alive:
                self.energy -= 1
                for i in range(8):
                    population[random.randint(0, len(population) - 1)].energize()
            return population
    
        # increment energy every time energy is received
        def energize(self):
            self.energy += 1
            if random.randint(1, 50) == 1:
                self.alive = False
    
        # reproduce if possible
        def reproduce(self, population, population_size):
            if self.energy > 10 and self.alive and len(population) < population_size:
                self.energy -= 10
                population.append(Organism(self.cooperator))
            return population
    

These methods make it very simple to write the rest of the genetic algorithm. While the previous script took a function-based approach, the functions in this script will mostly just iterate over the population and call these methods.
    
    
    # Create a population of objects
    def initialize(starting_cooperators, starting_noncooperators):
        population = []
        for i in range(starting_cooperators):
            population.append(Organism(True))
        for i in range(starting_noncooperators):
            population.append(Organism(False))
        return population
    

The position of organisms in the list does not matter as energy sharing is random. Thus, we simply initialize all of the cooperators and all of the non-cooperators in blocks.
    
    
    # fitness_function evaluates an individual
    def fitness_function(individual):
        return individual.energy
    
    # Adds one unit of energy to every organism
    def add_energy(population):
        for individual in population:
            individual.energize()
        return population
    
    # Let all cooperators cooperate
    def interact(population):
        for individual in population:
            population = individual.interact(population)
        return population
    
    # Remove dead organisms from the dish
    def clean_dish(population):
        for individual in population:
            if not individual.alive:
                population.remove(individual)
        return population
    
    # Let organisms reproduce if able
    def reproduce(population, maximum_population):
        reproduction_order = sorted(population, key=lambda x: fitness_function(x), reverse=True)
        for individual in reproduction_order:
            population = individual.reproduce(population, maximum_population)
        return population
    

Again, the actual functions of the genetic algorithm only need to iterate over the population and call class functions. The `clean_dish` function removes dead individuals from the population itself. The `reproduce` function begins by sorting organisms into a reproduction order based on their fitness: the most fit individuals get to reproduce first, and if space runs out before all organisms with enough energy can reproduce, those organisms must wait for others to die.
    
    
    # Handle the output of the Genetic Algorithm
    def termination(population, starting_cooperators, starting_noncooperators, maximum_population, total_iterations):
        print("Ran", total_iterations, "generations in a dish with a capacity of", maximum_population)
        print("Beginning Population:")
        print("C" * starting_cooperators + "N" * starting_noncooperators)
        print("Ending Population")
        population_string = ""
        for individual in population:
            if individual.cooperator:
                population_string += "C"
            else:
                population_string += "N"
        print(population_string)
    

The termination function prints the composition of the beginning and ending populations. The ending population will be outputted in the same order as it exists in the list, but remember that the order does not really matter.
    
    
    # Main function runs when script is called
    f __name__ == "__main__":
        if len(sys.argv) != 5:
            print("Usage: python petri.py coop noncoop max iter")
            exit()
        starting_cooperators = int(sys.argv[1])
        starting_noncooperators = int(sys.argv[2])
        maximum_population = int(sys.argv[3])
        if starting_cooperators + starting_noncooperators > maximum_population:
            print("maximum population less than starting population")
            exit()
        total_iterations = max(int(sys.argv[4]), 1) # must run for at least 1 iteration
        population = initialize(starting_cooperators, starting_noncooperators)
        for iteration in range(total_iterations):
            population = add_energy(population) # every organism generates energy
            population = interact(population) # energy is shared and consumed
            population = clean_dish(population) # remove dead organisms from the dish
            population = reproduce(population, maximum_population)
        termination(population, starting_cooperators, starting_noncooperators, maximum_population, total_iterations)
    

After reading and checking command line arguments, we initialize the appropriate starting population. For the desired number of iterations, we run the simulation: adding and sharing energy, removing dead organisms, and creating new ones. At termination, we output information about the final state of the population.  
Let's start by filling the dish to capacity with 10 cooperators and 10 non-cooperators and evolving 50 generations. Running `python code/petri.py 10 10 20 50` yields:
    
    
    Ran 50 generations in a dish with a capacity of 20
    Beginning Population:
    CCCCCCCCCCNNNNNNNNNN
    Ending Population
    NNNNNNNNNNNNNNNNNNNN
    

Run that a few times to see if the output changes. For me, most times the non-cooperators take over the system. Occasionally, there are a few cooperators left after 50 generations, and I had one time when the cooperators took over completely. As we can see, there is a massive variance in the outcome of this system, but the most likely result is all non-cooperators.  
Let's shake it up by increasing the dish size to 80 and the generations to 500. Keeping 10 cooperators and only putting one non-cooperator into the system yields a mix of results. Run `python code/petri.py 10 1 80 500`.
    
    
    Ran 500 generations in a dish with a capacity of 80
    Beginning Population:
    CCCCCCCCCCN
    Ending Population
    NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
    

This run shows that given enough time, even a single non-cooperator can take over the entire system. However, in this simulation, it is more likely than before that the cooperators take over entirely, which is guaranteed if the only non-cooperator dies before it can reproduce. After this many iterations, a mixed output is unlikely but still possible.

## A Framework to Build Upon

Now that we've considered two different types of genetic algorithms, you have the tools to apply them across multiple domains. Remember that genetic algorithms have limitations: they require a known function and known input types, only provide estimated answers, and can take a long time to run. However, current research around better selection and reproduction methods (linked below) promises to improve the approach for more real-world applications. In addition to the use cases that we've considered, genetic algorithms can be used for everything from controlling enemy attributes and behavior in a video game to selecting stocks in a financial model. I'll leave you with a few resources for further exploration.

  * [Sample code on FloydHub](https://floydhub.com/run?template=https://github.com/philipkiely/floydhub_genetic_algorithm_tutorial) includes a template to use as a starting point for your own genetic algorithms.
  * [Wikipedia](https://en.wikipedia.org/wiki/Genetic_algorithm) has a fairly complete theoretical overview.
  * [Essentials of Metaheuristics](https://cs.gmu.edu/~sean/book/metaheuristics/) is a free book by Sean Luke at George Mason University gives a more complete review of important algorithms in the field.

#### **FloydHub Call for AI writers**

Want to write amazing articles like Philip and play your role in the long road to Artificial General Intelligence? [We are looking for passionate writers](https://floydhub.github.io/write-for-floydhub/?utm_source=floydhub&utm_medium=banner&utm_campaign=call_for_writers_2019), to build the world's best blog for practical applications of groundbreaking A.I. techniques. FloydHub has a large reach within the AI community and with your help, we can inspire the next wave of AI. [Apply now](https://goo.gl/forms/PbOw0VmUnOfO1Lxp1) and join the crew!

**About Philip Kiely**

Philip Kiely writes code and words. He is the author of _Writing for Software Developers_ (2020). Philip holds a B.A. with honors in Computer Science from Grinnell College. Philip is a [FloydHub AI Writer](https://floydhub.github.io/write-for-floydhub/). You can find his work at[ https://philipkiely.com](https://philipkiely.com) or you can connect with Philip via[ LinkedIn](https://linkedin.com/in/philipkiely) and[ GitHub](https://github.com/philipkiely).