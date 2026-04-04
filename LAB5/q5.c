#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_VAL 10000
#define TAG_WORK  1
#define TAG_DONE  2

// Returns 1 if n is a perfect number, 0 otherwise
int is_perfect(int n) {
    if (n < 2) return 0;
    int sum = 1;  // 1 is always a proper divisor for n > 1
    for (int i = 2; (long long)i * i <= n; i++) {
        if (n % i == 0) {
            sum += i;
            if (i != n / i) sum += n / i;
        }
    }
    return (sum == n) ? 1 : 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        fprintf(stderr, "Need at least 2 processes (1 master + 1 slave).\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        // ---- MASTER ----
        int next_num = 2;
        int active_slaves = size - 1;
        int msg;
        MPI_Status status;
        printf("Perfect numbers up to %d:\n", MAX_VAL);

        while (active_slaves > 0) {
            MPI_Recv(&msg, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG,
                     MPI_COMM_WORLD, &status);
            int slave = status.MPI_SOURCE;

            // Interpret reply: 0 = starting, positive = perfect, negative = not perfect
            if (msg > 0) {
                printf("  PERFECT: %d\n", msg);
            }
            // msg == 0: slave just started; msg < 0: not perfect — nothing to print

            if (next_num <= MAX_VAL) {
                MPI_Send(&next_num, 1, MPI_INT, slave, TAG_WORK, MPI_COMM_WORLD);
                next_num++;
            } else {
                int sentinel = -1;
                MPI_Send(&sentinel, 1, MPI_INT, slave, TAG_DONE, MPI_COMM_WORLD);
                active_slaves--;
            }
        }
        printf("Done.\n");

    } else {
        // ---- SLAVE ----
        int num_to_test;
        MPI_Status status;

        // Signal start with 0
        int zero = 0;
        MPI_Send(&zero, 1, MPI_INT, 0, TAG_WORK, MPI_COMM_WORLD);

        while (1) {
            MPI_Recv(&num_to_test, 1, MPI_INT, 0, MPI_ANY_TAG,
                     MPI_COMM_WORLD, &status);
            if (status.MPI_TAG == TAG_DONE) break;

            int result = is_perfect(num_to_test) ? num_to_test : -num_to_test;
            MPI_Send(&result, 1, MPI_INT, 0, TAG_WORK, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
