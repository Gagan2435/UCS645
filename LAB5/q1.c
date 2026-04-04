#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N (1 << 16)   // 2^16 = 65536

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double a = 2.5;
    double *X = (double*)malloc(N * sizeof(double));
    double *Y = (double*)malloc(N * sizeof(double));

    // Initialize on all ranks (simple fill)
    for (int i = 0; i < N; i++) {
        X[i] = (double)i;
        Y[i] = (double)(N - i);
    }

    // --- Uniprocessor baseline (only rank 0 measures) ---
    double seq_time = 0.0;
    if (rank == 0) {
        double *Xs = (double*)malloc(N * sizeof(double));
        double *Ys = (double*)malloc(N * sizeof(double));
        for (int i = 0; i < N; i++) { Xs[i] = (double)i; Ys[i] = (double)(N-i); }
        double t0 = MPI_Wtime();
        for (int i = 0; i < N; i++) Xs[i] = a * Xs[i] + Ys[i];
        seq_time = MPI_Wtime() - t0;
        printf("Sequential DAXPY time: %.6f seconds\n", seq_time);
        free(Xs); free(Ys);
    }

    // --- MPI parallel DAXPY ---
    int chunk = N / size;
    int start = rank * chunk;
    int end   = (rank == size - 1) ? N : start + chunk;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int i = start; i < end; i++)
        X[i] = a * X[i] + Y[i];

    // Gather isn't strictly needed for timing, but ensures correctness
    double par_time = MPI_Wtime() - t0;

    // Max time across all ranks
    double max_par_time;
    MPI_Reduce(&par_time, &max_par_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Parallel DAXPY time (%d procs): %.6f seconds\n", size, max_par_time);
        if (max_par_time > 0)
            printf("Speedup: %.2fx\n", seq_time / max_par_time);
    }

    free(X); free(Y);
    MPI_Finalize();
    return 0;
}
