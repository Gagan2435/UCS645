#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ARRAY_SIZE 1000000

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int chunk = ARRAY_SIZE / size;
    int *array = NULL;
    int *local_array = (int*)malloc(chunk * sizeof(int));

    if (rank == 0) {
        array = (int*)malloc(ARRAY_SIZE * sizeof(int));
        for (int i = 0; i < ARRAY_SIZE; i++) array[i] = i + 1;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    MPI_Scatter(array, chunk, MPI_INT, local_array, chunk, MPI_INT, 0, MPI_COMM_WORLD);

    long long local_sum = 0;
    for (int i = 0; i < chunk; i++) local_sum += local_array[i];

    long long global_sum = 0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double total_time = MPI_Wtime() - start;

    if (rank == 0) {
        printf("Processes=%d | Time=%.6f seconds | Sum=%lld\n", size, total_time, global_sum);
        if (array) free(array);
    }

    free(local_array);
    MPI_Finalize();
    return 0;
}
