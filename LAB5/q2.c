#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ARRAY_SIZE 10000000  // 10 million doubles (~80 MB)

// Custom broadcast: Rank 0 sends to each rank one by one (linear)
void MyBcast(double *buf, int count, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == 0) {
        for (int dest = 1; dest < size; dest++)
            MPI_Send(buf, count, MPI_DOUBLE, dest, 0, comm);
    } else {
        MPI_Recv(buf, count, MPI_DOUBLE, 0, 0, comm, MPI_STATUS_IGNORE);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double *buf = (double*)malloc(ARRAY_SIZE * sizeof(double));

    // Initialize on rank 0
    if (rank == 0)
        for (int i = 0; i < ARRAY_SIZE; i++) buf[i] = (double)i;

    // ---- Part A: MyBcast (linear for-loop) ----
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    MyBcast(buf, ARRAY_SIZE, MPI_COMM_WORLD);
    double my_time = MPI_Wtime() - t0;

    double max_my;
    MPI_Reduce(&my_time, &max_my, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Reset buffer
    if (rank != 0)
        for (int i = 0; i < ARRAY_SIZE; i++) buf[i] = 0.0;

    // ---- Part B: MPI_Bcast (optimised tree-based) ----
    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    MPI_Bcast(buf, ARRAY_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double mpi_time = MPI_Wtime() - t1;

    double max_mpi;
    MPI_Reduce(&mpi_time, &max_mpi, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Processes: %d\n", size);
        printf("MyBcast  time: %.6f seconds\n", max_my);
        printf("MPI_Bcast time: %.6f seconds\n", max_mpi);
        printf("Speedup (MPI_Bcast vs MyBcast): %.2fx\n",
               (max_my > 0) ? max_my / max_mpi : 0.0);
        printf("\nAnalysis:\n");
        printf("  MyBcast is O(p) - linear: Rank 0 sends %d times sequentially.\n", size-1);
        printf("  MPI_Bcast is O(log p) - tree-based: halves remaining receivers each step.\n");
    }

    free(buf);
    MPI_Finalize();
    return 0;
}
