#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ARRAY_SIZE 100

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
        printf("Initializing array with values 1 to 100...\n");
        for (int i = 0; i < ARRAY_SIZE; i++) {
            array[i] = i + 1;
        }
    }

    MPI_Scatter(array, chunk, MPI_INT, local_array, chunk, MPI_INT, 0, MPI_COMM_WORLD);

    int local_sum = 0;
    for (int i = 0; i < chunk; i++) {
        local_sum += local_array[i];
    }
    printf("Process %d: local sum = %d\n", rank, local_sum);

    int global_sum = 0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nGlobal Sum = %d\n", global_sum);
        printf("Expected   = 5050\n");
        printf("Average    = %.2f\n", (double)global_sum / ARRAY_SIZE);
        free(array);
    }

    free(local_array);
    MPI_Finalize();
    return 0;
}
