package functions;

import java.util.Arrays;

public class BenchmarkFunctions {

    public static double ackley(double[] x, double a, double b, double c) {
        // Check for NaN or Infinity values
        if (Arrays.stream(x).anyMatch(Double::isNaN) || Arrays.stream(x).anyMatch(Double::isInfinite)) {
            throw new IllegalArgumentException("Input contains NaN or Infinite values.");
        }

        int n = x.length;
        double s1 = 0.0;
        double s2 = 0.0;

        // Compute the summation terms
        for (double xi : x) {
            s1 += Math.pow(xi, 2); // Sum of squares
            s2 += Math.cos(c * xi); // Sum of cosines
        }

        // Calculate the Ackley function
        double term1 = -a * Math.exp(-b * Math.sqrt(s1 / n));
        double term2 = -Math.exp(s2 / n);

        return term1 + term2 + a + Math.exp(1);
    }
    // Overloaded method with default parameters (a=20, b=0.2, c=2π)
    public static double ackley(double[] x) {
        return ackley(x, 20, 0.2, 2 * Math.PI);
    }


    public static double dixonPrice(double[] x) {
        // Check for NaN or Infinity values
        if (Arrays.stream(x).anyMatch(Double::isNaN) || Arrays.stream(x).anyMatch(Double::isInfinite)) {
            throw new IllegalArgumentException("Input contains NaN or Infinite values.");
        }

        int n = x.length;
        if (n < 2) {
            throw new IllegalArgumentException("Input array must have at least 2 elements.");
        }

        double result = Math.pow(x[0] - 1, 2); // First term: (x[0] - 1)^2

        // Compute the summation terms
        for (int j = 2; j <= n; j++) {
            double term = j * Math.pow(2 * Math.pow(x[j - 1], 2) - x[j - 2], 2);
            result += term;
        }

        return result;
    }


    public static double griewank(double[] x, double fr) {
        // Check for NaN or Infinity values
        if (Arrays.stream(x).anyMatch(Double::isNaN) || Arrays.stream(x).anyMatch(Double::isInfinite)) {
            throw new IllegalArgumentException("Input contains NaN or Infinite values.");
        }

        int n = x.length;
        double sum = 0.0;
        double product = 1.0;

        // Compute the summation and product terms
        for (int i = 0; i < n; i++) {
            sum += Math.pow(x[i], 2); // Sum of squares
            product *= Math.cos(x[i] / Math.sqrt(i + 1)); // Product of cos(x[i] / sqrt(i + 1))
        }

        return sum / fr - product + 1;
    }
    // Overloaded method with default fr = 4000
    public static double griewank(double[] x) {
        return griewank(x, 4000);
    }


    public static double perm(double[] x, double b) {
        // Check for NaN or Infinity values
        if (Arrays.stream(x).anyMatch(Double::isNaN) || Arrays.stream(x).anyMatch(Double::isInfinite)) {
            throw new IllegalArgumentException("Input contains NaN or Infinite values.");
        }

        int n = x.length;
        double result = 0.0;

        // Outer loop over k values
        for (int k = 1; k <= n; k++) {
            double innerSum = 0.0;

            // Inner loop over j values
            for (int j = 1; j <= n; j++) {
                double term = Math.pow(j, k) + b;
                double xByJ = Math.abs(x[j - 1]) / j;
                innerSum += term * (Math.pow(xByJ, k) - 1);
            }

            // Square the inner sum and add to the result
            result += Math.pow(innerSum, 2);
        }

        // Return the final result
        return result / n; // Averaging as implied by the Python function
    }
    // Overloaded method with default b = 0.5
    public static double perm(double[] x) {
        return perm(x, 0.5);
    }


    public static double rastrigin(double[] x) {
        // Check for NaN or Infinity values
        if (Arrays.stream(x).anyMatch(Double::isNaN) || Arrays.stream(x).anyMatch(Double::isInfinite)) {
            throw new IllegalArgumentException("Input contains NaN or Infinite values.");
        }

        int n = x.length;
        double result = 10 * n; // The constant term: 10 * n

        // Compute the summation term
        for (double xi : x) {
            result += Math.pow(xi, 2) - 10 * Math.cos(2 * Math.PI * xi);
        }

        return result;
    }


    public static double rosenbrock(double[] x) {
        // Check for NaN or Infinity values
        if (Arrays.stream(x).anyMatch(Double::isNaN) || Arrays.stream(x).anyMatch(Double::isInfinite)) {
            throw new IllegalArgumentException("Input contains NaN or Infinite values.");
        }

        int n = x.length;
        if (n < 2) {
            throw new IllegalArgumentException("Input array must have at least 2 elements.");
        }

        double result = 0.0;

        // Compute the Rosenbrock function
        for (int i = 0; i < n - 1; i++) {
            double x0 = x[i];
            double x1 = x[i + 1];
            result += Math.pow(1 - x0, 2) + 100 * Math.pow(x1 - Math.pow(x0, 2), 2);
        }

        return result;
    }


    public static double schwefel(double[] x) {
        // Check for NaN or Infinity values
        if (Arrays.stream(x).anyMatch(Double::isNaN) || Arrays.stream(x).anyMatch(Double::isInfinite)) {
            throw new IllegalArgumentException("Input contains NaN or Infinite values.");
        }

        int n = x.length;
        double result = 418.9829 * n; // The constant term: 418.9829 * n

        // Compute the summation term
        for (double xi : x) {
            result -= xi * Math.sin(Math.sqrt(Math.abs(xi)));
        }

        return result;
    }


    public static double sphere(double[] x) {
        // Check for NaN or Infinity values
        if (Arrays.stream(x).anyMatch(Double::isNaN) || Arrays.stream(x).anyMatch(Double::isInfinite)) {
            throw new IllegalArgumentException("Input contains NaN or Infinite values.");
        }

        // Compute the summation of squares
        double result = 0.0;
        for (double xi : x) {
            result += Math.pow(xi, 2);
        }

        return result;
    }


    public static double zakharov(double[] x) {
        // Check for NaN or Infinity values
        if (Arrays.stream(x).anyMatch(Double::isNaN) || Arrays.stream(x).anyMatch(Double::isInfinite)) {
            throw new IllegalArgumentException("Input contains NaN or Infinite values.");
        }

        int n = x.length;
        double result = 0.0;

        // Sum of squares term
        for (double xi : x) {
            result += Math.pow(xi, 2);
        }

        // Sum of x[i] * i / 2 (j is effectively the index in the original function)
        double s2 = 0.0;
        for (int i = 0; i < n; i++) {
            s2 += (i + 1) * x[i];
        }
        s2 /= 2;

        // Add the s2^2 and s2^4 terms
        result += Math.pow(s2, 2) + Math.pow(s2, 4);

        return result;
    }
}
