package algorithms;

import java.util.*;


public class MothFlameOptimizer {
    private int populationSize;
    private int dimension;
    private int maxIterations;
    private double[] lowerBound;
    private double[] upperBound;
    private Random random;

    public MothFlameOptimizer(int populationSize, int dimension, int maxIterations,
                              double[] lowerBound, double[] upperBound) {
        this.populationSize = populationSize;
        this.dimension = dimension;
        this.maxIterations = maxIterations;
        this.lowerBound = lowerBound;
        this.upperBound = upperBound;
        this.random = new Random();
    }

    public double[] optimize(ObjectiveFunction function) {
        // Initialize moths and flames
        double[][] moths = initializePopulation();
        double[][] flames = new double[populationSize][dimension];
        double[] mothFitness = new double[populationSize];
        double[] flameFitness = new double[populationSize];

        // Best solution tracking
        double[] bestSolution = null;
        double bestFitness = Double.MAX_VALUE;

        // Evaluate initial population in parallel
        Arrays.parallelSetAll(moths, i -> {
            mothFitness[i] = function.evaluate(moths[i]);
            return moths[i];
        });
        bestSolution = getBestSolution(moths, mothFitness);
        bestFitness = function.evaluate(bestSolution);

        // Main optimization loop
        for (int iteration = 1; iteration <= maxIterations; iteration++) {
            int flameCount = Math.round(populationSize - iteration * ((float) populationSize - 1) / maxIterations);

            // Sort moths and update flames (this might still need to be done serially)
            sortAndUpdateFlames(moths, mothFitness, flames, flameFitness);

            // Calculate the parameter 'a' based on the iteration
            double a = -1 + iteration * (-1.0 / maxIterations);

            // Update moth positions in parallel
            Arrays.parallelSetAll(moths, i -> {
                for (int j = 0; j < dimension; j++) {
                    int flameIndex = i;
                    if (i >= flameCount) {
                        flameIndex = flameCount - 1;
                    }

                    double distance = Math.abs(flames[flameIndex][j] - moths[i][j]);
                    double b = 1.0;
                    double t = (a - 1) * random.nextDouble() + 1;
                    double newPosition = distance * Math.exp(b * t) * Math.cos(2 * Math.PI * t) + flames[flameIndex][j];

                    moths[i][j] = Math.max(lowerBound[j], Math.min(upperBound[j], newPosition));
                }

                double newFitness = function.evaluate(moths[i]);
                mothFitness[i] = newFitness;

                return moths[i];
            });
            // Update the best solution
            bestSolution = getBestSolution(moths, mothFitness);
            bestFitness = function.evaluate(bestSolution);
        }

        return bestSolution;
    }

    private double[] getBestSolution(double[][] moths, double[] mothFitness) {
        double bestFitness = Double.MAX_VALUE;
        double[] bestSolution = null;

        for (int i = 0; i < populationSize; i++) {
            if (mothFitness[i] < bestFitness) {
                bestFitness = mothFitness[i];
                bestSolution = moths[i].clone();
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

    private void sortAndUpdateFlames(double[][] moths, double[] mothFitness,
                                     double[][] flames, double[] flameFitness) {
        // Create indices array
        Integer[] indices = new Integer[populationSize];
        for (int i = 0; i < populationSize; i++) {
            indices[i] = i;
        }

        // Sort indices based on fitness
        Arrays.sort(indices, (a, b) -> Double.compare(mothFitness[a], mothFitness[b]));

        // Update flames and flame fitness
        for (int i = 0; i < populationSize; i++) {
            flames[i] = moths[indices[i]].clone();
            flameFitness[i] = mothFitness[indices[i]];
        }
    }

    // Interface for objective function
    public interface ObjectiveFunction {
        double evaluate(double[] solution);
    }
}

