package results;

public class Result {
    private final String optimizerName;
    private final String benchmarkName;
    private final int populationSize;
    private final int generations;
    private final double bestFitness;
    private final double avgFitness;
    private final double stdDev;
    private final long evalCount;

    public Result(String optimizerName, String benchmarkName, int populationSize, int generations,
                  double bestFitness, double avgFitness, double stdDev, long evalCount) {
        this.optimizerName = optimizerName;
        this.benchmarkName = benchmarkName;
        this.populationSize = populationSize;
        this.generations = generations;
        this.bestFitness = bestFitness;
        this.avgFitness = avgFitness;
        this.stdDev = stdDev;
        this.evalCount = evalCount;
    }

    // Getters for CSV writing
    public String toCsv() {
        return String.join(",",
                optimizerName, benchmarkName, String.valueOf(populationSize), String.valueOf(generations),
                String.format("%.6f", bestFitness), String.format("%.6f", avgFitness),
                String.format("%.6f", stdDev), String.valueOf(evalCount)
        );
    }

    public static String csvHeader() {
        return "Optimizer,Benchmark,Population Size,Generations,Best Fitness,Average Fitness,StdDev,Eval Count";
    }
}

