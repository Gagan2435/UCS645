#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define TOTAL_SIZE 500000000LL  // 500 million elements

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Step 1: Rank 0 gets multiplier, broadcasts it
    double multiplier = 1.0;
    if (rank == 0) {
        printf("Enter scaling multiplier (e.g. 1.0): ");
        fflush(stdout);
        if (scanf("%lf", &multiplier) != 1) multiplier = 1.0;
    }
    MPI_Bcast(&multiplier, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Step 2: Each process generates its own local chunk
    long long chunk = TOTAL_SIZE / size;
    long long remainder = TOTAL_SIZE % size;
    // Last rank absorbs remainder
    long long local_size = (rank == size - 1) ? chunk + remainder : chunk;

    double *A = (double*)malloc(local_size * sizeof(double));
    double *B = (double*)malloc(local_size * sizeof(double));
    if (!A || !B) {
        fprintf(stderr, "Rank %d: malloc failed for local_size=%lld\n", rank, local_size);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (long long i = 0; i < local_size; i++) {
        A[i] = 1.0;
        B[i] = 2.0 * multiplier;
    }

    // Step 3: Local dot product
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    double local_dot = 0.0;
    for (long long i = 0; i < local_size; i++)
        local_dot += A[i] * B[i];

    // Step 4: Global reduction
    double final_result = 0.0;
    MPI_Reduce(&local_dot, &final_result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double elapsed = MPI_Wtime() - t0;
    double max_time;
    MPI_Reduce(&elapsed, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Processes      : %d\n", size);
        printf("Dot product    : %.2f\n", final_result);
        printf("Time taken     : %.6f seconds\n", max_time);
        printf("\n(Run with 1, 2, 4, 8 processes and record times for speedup analysis)\n");
        printf("Speedup = T(1) / T(N),  Efficiency = Speedup / N\n");
    }

    free(A); free(B);
    MPI_Finalize();
    return 0;
}
