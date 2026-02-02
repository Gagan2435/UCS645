#include <stdio.h>
#include <omp.h>

int main() {

    long num_steps = 10000000;   // 10 million steps
    double step;
    double sum = 0.0;
    double x, pi;

    step = 1.0 / (double) num_steps;

    double start = omp_get_wtime();

    // Parallel loop with reduction
    #pragma omp parallel for reduction(+:sum)
    for(long i = 0; i < num_steps; i++) {

        x = (i + 0.5) * step;
        sum = sum + 4.0 / (1.0 + x * x);

    }

    pi = step * sum;

    double end = omp_get_wtime();

    printf("Calculated PI = %f\n", pi);
    printf("Time Taken = %f seconds\n", end - start);

    return 0;
}
