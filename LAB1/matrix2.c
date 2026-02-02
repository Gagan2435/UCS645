#include <stdio.h>
#include <omp.h>

#define N 300   // Matrix size

int A[N][N], B[N][N], C[N][N];

int main() {

    int i, j, k;

    // Initialize matrices
    for(i = 0; i < N; i++) {
        for(j = 0; j < N; j++) {
            A[i][j] = 1;
            B[i][j] = 1;
            C[i][j] = 0;
        }
    }

    double start = omp_get_wtime();

    // 2D Parallelization using collapse
    #pragma omp parallel for collapse(2)
    for(i = 0; i < N; i++) {
        for(j = 0; j < N; j++) {

            int sum = 0;

            for(k = 0; k < N; k++) {
                sum += A[i][k] * B[k][j];
            }

            C[i][j] = sum;
        }
    }

    double end = omp_get_wtime();

    printf("Matrix Multiplication (2D Parallel) Done\n");
    printf("Time Taken = %f seconds\n", end - start);

    return 0;
}
