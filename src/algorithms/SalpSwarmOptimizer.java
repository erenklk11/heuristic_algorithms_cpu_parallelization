package algorithms;

import java.util.*;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.IntStream;

public class SalpSwarmOptimizer {
    private int populationSize;
    private int dimension;
    private int maxIterations;
    private double[] lowerBound;
    private double[] upperBound;
    private Random random;

    public SalpSwarmOptimizer(int populationSize, int dimension, int maxIterations,
                              double[] lowerBound, double[] upperBound) {
        this.populationSize = populationSize;
        this.dimension = dimension;
        this.maxIterations = maxIterations;
        this.lowerBound = lowerBound;
        this.upperBound = upperBound;
        this.random = new Random();
    }

    public double[] optimize(ObjectiveFunction function) {
        // Initialize salp population
        double[][] salps = initializePopulation();
        double[] fitness = new double[populationSize];

// Evaluate initial population in parallel
        IntStream.range(0, populationSize).parallel().forEach(i -> {
            fitness[i] = function.evaluate(salps[i]);
        });

        // Find initial food source (best solution)
        int bestIndex = getBestIndex(fitness);
        double[] foodSource = salps[bestIndex].clone();
        double bestFitness = fitness[bestIndex];

        // Main optimization loop
        for (int iteration = 0; iteration < maxIterations; iteration++) {
            // Calculate c1 parameter (coefficient of exploration/exploitation)
            double c1 = 2 * Math.exp(-Math.pow(4 * iteration / maxIterations, 2));

            // Update leader salp (first salp)
            for (int j = 0; j < dimension; j++) {
                double c2 = random.nextDouble();
                double c3 = random.nextDouble();

                // Update leader's position
                if (c3 >= 0.5) {
                    salps[0][j] = foodSource[j] + c1 * ((upperBound[j] - lowerBound[j]) * c2 + lowerBound[j]);
                } else {
                    salps[0][j] = foodSource[j] - c1 * ((upperBound[j] - lowerBound[j]) * c2 + lowerBound[j]);
                }

                // Boundary check for leader
                salps[0][j] = Math.max(lowerBound[j], Math.min(upperBound[j], salps[0][j]));
            }

            // Update follower salps in parallel
            IntStream.range(1, populationSize).parallel().forEach(i -> {
                for (int j = 0; j < dimension; j++) {
                    double velocity = (salps[i][j] + salps[i-1][j]) / 2;
                    salps[i][j] = velocity;

                    salps[i][j] = Math.max(lowerBound[j], Math.min(upperBound[j], salps[i][j]));
                }
            });

// Create atomic references for thread-safe updates
            AtomicReference<Double> bestFitnessRef = new AtomicReference<>(bestFitness);
            AtomicReference<double[]> foodSourceRef = new AtomicReference<>(foodSource);

// Evaluate new positions and update food source in parallel
            IntStream.range(0, populationSize).parallel().forEach(i -> {
                fitness[i] = function.evaluate(salps[i]);
                synchronized (this) {
                    if (fitness[i] < bestFitnessRef.get()) {
                        bestFitnessRef.set(fitness[i]);
                        foodSourceRef.set(salps[i].clone());
                    }
                }
            });
// Update the main variables after parallel execution
            bestFitness = bestFitnessRef.get();
            foodSource = foodSourceRef.get();

            // Optional: Add diversity mechanism for stagnation
            if (iteration % 20 == 0) {
                addDiversity(salps, foodSource);
            }
        }

        return foodSource;
    }

    private double[][] initializePopulation() {
        double[][] population = new double[populationSize][dimension];
        for (int i = 0; i < populationSize; i++) {
            for (int j = 0; j < dimension; j++) {
                population[i][j] = lowerBound[j] +
                        random.nextDouble() * (upperBound[j] - lowerBound[j]);
            }
        }
        return population;
    }

    private int getBestIndex(double[] fitness) {
        int bestIdx = 0;
        for (int i = 1; i < fitness.length; i++) {
            if (fitness[i] < fitness[bestIdx]) {
                bestIdx = i;
            }
        }
        return bestIdx;
    }

    private void addDiversity(double[][] salps, double[] foodSource) {
        // Randomly reinitialize some followers while keeping the best solution
        for (int i = 1; i < populationSize; i++) {
            if (random.nextDouble() < 0.1) { // 10% chance to reinitialize
                for (int j = 0; j < dimension; j++) {
                    salps[i][j] = lowerBound[j] +
                            random.nextDouble() * (upperBound[j] - lowerBound[j]);
                }
            }
        }
    }

    // Interface for objective function
    public interface ObjectiveFunction {
        double evaluate(double[] solution);
    }
}
