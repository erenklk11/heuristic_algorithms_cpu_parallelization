import algorithms.GoldenEagleOptimizer;
import algorithms.MothFlameOptimizer;
import algorithms.ObjectiveFunction;
import algorithms.SalpSwarmOptimizer;

import java.util.*;
import java.util.concurrent.*;

import functions.BenchmarkFunctions;


public class Main {

    public static void main(String[] args) throws InterruptedException, ExecutionException {
        // Configuration for optimization
        int dimension = 30;
        double[] lowerBound = new double[dimension];
        double[] upperBound = new double[dimension];
        Arrays.fill(lowerBound, -100);
        Arrays.fill(upperBound, 100);

        // Population size variations for parameter optimization
        int[] populationSizes = {50, 100, 200, 500};
        int[] numOfGenerations = {100, 250, 500, 1000};

        // List of benchmark functions with names
        List<Map.Entry<String, ObjectiveFunction>> benchmarkFunctions = Arrays.asList(
                Map.entry("Ackley", BenchmarkFunctions::ackley),
                Map.entry("DixonPrice", BenchmarkFunctions::dixonPrice),
                Map.entry("Griewank", BenchmarkFunctions::griewank),
                Map.entry("Perm", BenchmarkFunctions::perm),
                Map.entry("Rastrigin", BenchmarkFunctions::rastrigin),
                Map.entry("Rosenbrock", BenchmarkFunctions::rosenbrock),
                Map.entry("Schwefel", BenchmarkFunctions::schwefel),
                Map.entry("Sphere", BenchmarkFunctions::sphere),
                Map.entry("Zakharov", BenchmarkFunctions::zakharov)
        );

        // List of optimization algorithms
        List<Class<?>> optimizers = Arrays.asList(
                SalpSwarmOptimizer.class,
                MothFlameOptimizer.class,
                GoldenEagleOptimizer.class
        );

        // Parallel execution of benchmark functions
        ForkJoinPool pool = new ForkJoinPool(Runtime.getRuntime().availableProcessors());
        List<Callable<String>> tasks = new ArrayList<>();

        for (Map.Entry<String, ObjectiveFunction> benchmarkEntry : benchmarkFunctions) {
            for (Class<?> optimizerClass : optimizers) {
                for (int populationSize : populationSizes) {
                    for (int noOfGeneration : numOfGenerations) {
                        tasks.add(() -> {
                            String threadName = Thread.currentThread().getName();
                            String benchmarkName = benchmarkEntry.getKey();
                            System.out.printf(
                                    "Thread: %s is optimizing %s using %s with Population Size: %d, Generations: %d%n",
                                    threadName, benchmarkName, optimizerClass.getSimpleName(), populationSize, noOfGeneration
                            );

                            return runOptimizer(optimizerClass, benchmarkEntry.getValue(), populationSize, dimension, noOfGeneration, lowerBound, upperBound);
                        });
                    }
                }
            }
        }

        List<Future<String>> results = pool.invokeAll(tasks);
        pool.shutdown();

        // Print the collected results
        System.out.println("\nFinal Results:");
        for (Future<String> result : results) {
            try {
                System.out.println(result.get());
            } catch (Exception e) {
                System.err.println("Task failed: " + e.getMessage());
                e.printStackTrace();
            }
        }
    }

    private static String runOptimizer(Class<?> optimizerClass, ObjectiveFunction benchmark, int populationSize,
                                       int dimension, int maxIterations, double[] lowerBound, double[] upperBound) {
        // Run the algorithm 5 times for robustness evaluation
        int runs = 5;
        double[] fitnesses = new double[runs];
        long evaluations = 0;

        for (int run = 0; run < runs; run++) {
            try {
                Object optimizer = optimizerClass.getConstructor(int.class, int.class, int.class, double[].class, double[].class)
                        .newInstance(populationSize, dimension, maxIterations, lowerBound, upperBound);

                double[] bestSolution;
                if (optimizer instanceof SalpSwarmOptimizer) {
                    bestSolution = ((SalpSwarmOptimizer) optimizer).optimize(benchmark);
                    evaluations += ((SalpSwarmOptimizer) optimizer).getEvaluationCount();
                } else if (optimizer instanceof MothFlameOptimizer) {
                    bestSolution = ((MothFlameOptimizer) optimizer).optimize(benchmark);
                    evaluations += ((MothFlameOptimizer) optimizer).getEvaluationCount();
                } else if (optimizer instanceof GoldenEagleOptimizer) {
                    bestSolution = ((GoldenEagleOptimizer) optimizer).optimize(benchmark);
                    evaluations += ((GoldenEagleOptimizer) optimizer).getEvaluationCount();
                } else {
                    throw new IllegalArgumentException("Unknown optimizer.");
                }

                fitnesses[run] = benchmark.evaluate(bestSolution);
            } catch (Exception e) {
                System.err.println("Error in optimizer: " + optimizerClass.getSimpleName());
                e.printStackTrace();
                return "Error occurred";
            }
        }

        // Compute statistics
        double best = Arrays.stream(fitnesses).min().orElse(Double.MAX_VALUE);
        double avg = Arrays.stream(fitnesses).average().orElse(Double.MAX_VALUE);
        double stdDev = Math.sqrt(Arrays.stream(fitnesses).map(f -> Math.pow(f - avg, 2)).average().orElse(0.0));

        return String.format(
                "Optimizer: %s, Population: %d, Best: %.6f, Avg: %.6f, StdDev: %.6f, EvalCount: %d",
                optimizerClass.getSimpleName(), populationSize, best, avg, stdDev, evaluations
        );
    }
}

