import random
import numpy
import math
from solution import solution

def GEA(objf, lb, ub, dim, N, Max_iteration):
    # Initialize population
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    Eagles = numpy.zeros((N, dim))
    for i in range(dim):
        Eagles[:, i] = numpy.random.uniform(0, 1, N) * (ub[i] - lb[i]) + lb[i]
    Fitness = numpy.full(N, float("inf"))
    
    Convergence_curve = numpy.zeros(Max_iteration)
    
    s = solution()
        
    
    # Evaluate initial fitness
    for i in range(0, N):
        Fitness[i] = objf(Eagles[i, :])
    
    # Get best initial fitness and position
    BestFitness = numpy.min(Fitness)
    BestEagle = Eagles[numpy.argmin(Fitness), :]

    Iteration = 1
    
    while Iteration < Max_iteration:
        # Update position of each eagle
        for i in range(N):
            # Random exploration factor
            R = numpy.random.uniform(0, 1, dim)
            if numpy.random.rand() < 0.5:
                Eagles[i, :] = Eagles[i, :] + R * (BestEagle - Eagles[i, :])
            else:
                Eagles[i, :] = Eagles[i, :] + R * (Eagles[numpy.random.randint(0, N), :] - Eagles[i, :])
            
            # Ensure boundaries
            Eagles[i, :] = numpy.clip(Eagles[i, :], lb, ub)
            
            # Evaluate fitness
            Fitness[i] = objf(Eagles[i, :])
            
            # Update best position and fitness
            if Fitness[i] < BestFitness:
                BestFitness = Fitness[i]
                BestEagle = Eagles[i, :]

        # Log convergence
        Convergence_curve[Iteration] = BestFitness
        
        # Display progress
        if Iteration % 1 == 0:
            print(f"GEA: At iteration {Iteration}, the best fitness is {BestFitness}")
        
        Iteration += 1
    
    s.convergence = Convergence_curve
    s.optimizer = "GEA"
    s.bestIndividual = BestEagle

    return s
