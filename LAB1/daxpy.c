#include <stdio.h>
#include <omp.h>

#define N 65536   // Size = 2^16

int main() {

    double X[N], Y[N];
    double a = 2.0;
    int i;

    // Initialize arrays
    for(i = 0; i < N; i++) {
        X[i] = i * 1.0;
        Y[i] = i * 2.0;
    }

    double start = omp_get_wtime();

    // Parallel loop
    #pragma omp parallel for
    for(i = 0; i < N; i++) {
        X[i] = a * X[i] + Y[i];
    }

    double end = omp_get_wtime();

    printf("DAXPY Completed\n");
    printf("Time Taken = %f seconds\n", end - start);

    return 0;
}
