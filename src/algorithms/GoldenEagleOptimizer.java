package algorithms;

import java.util.*;
import java.util.stream.IntStream;


public class GoldenEagleOptimizer {
    private int populationSize;
    private int dimension;
    private int maxIterations;
    private double[] lowerBound;
    private double[] upperBound;
    private Random random;

    public GoldenEagleOptimizer(int populationSize, int dimension, int maxIterations,
                                double[] lowerBound, double[] upperBound) {
        this.populationSize = populationSize;
        this.dimension = dimension;
        this.maxIterations = maxIterations;
        this.lowerBound = lowerBound;
        this.upperBound = upperBound;
        this.random = new Random();
    }

    public double[] optimize(ObjectiveFunction function) {
        // Initialize population
        double[][] population = initializePopulation();
        double[] fitness = new double[populationSize];

        // Find initial best solution using parallel stream for fitness evaluation
        IntStream.range(0, populationSize).parallel().forEach(i -> {
            fitness[i] = function.evaluate(population[i]);
        });
        int bestIndex = getBestIndex(fitness);
        double[] bestSolution = population[bestIndex].clone();
        double bestFitness = fitness[bestIndex];

        // Main optimization loop
        for (int iteration = 0; iteration < maxIterations; iteration++) {
            double w = 0.5 - ((double) iteration / maxIterations) * 0.5; // Inertia weight

            for (int i = 0; i < populationSize; i++) {
                if (i != bestIndex) {
                    // Update each eagle's position
                    double[] newPosition = new double[dimension];

                    for (int j = 0; j < dimension; j++) {
                        // Golden eagle movement formula
                        double r1 = random.nextDouble();
                        double r2 = random.nextDouble();
                        double r3 = random.nextDouble();

                        // Social influence
                        double socialInfluence = r1 * (bestSolution[j] - population[i][j]);

                        // Exploration component
                        double exploration = r2 * (upperBound[j] - lowerBound[j]) * (2 * r3 - 1);

                        // Update position
                        newPosition[j] = population[i][j] + w * socialInfluence + (1 - w) * exploration;

                        // Boundary check
                        newPosition[j] = Math.max(lowerBound[j], Math.min(upperBound[j], newPosition[j]));
                    }

                    // Evaluate new position
                    double newFitness = function.evaluate(newPosition);

                    // Update if better
                    if (newFitness < fitness[i]) {
                        population[i] = newPosition;
                        fitness[i] = newFitness;

                        // Update global best if necessary
                        if (newFitness < bestFitness) {
                            bestSolution = newPosition.clone();
                            bestFitness = newFitness;
                            bestIndex = i;
                        }
                    }
                }
            }
        }

        return bestSolution;
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

    // Interface for objective function
    public interface ObjectiveFunction {
        double evaluate(double[] solution);
    }

}