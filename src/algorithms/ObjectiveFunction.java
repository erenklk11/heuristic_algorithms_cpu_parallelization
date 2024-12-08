package algorithms;

@FunctionalInterface
public interface ObjectiveFunction {
    double evaluate(double[] solution);
}
