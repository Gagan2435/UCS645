#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_VAL 10000
#define TAG_WORK  1
#define TAG_DONE  2

// Returns 1 if n is prime, 0 otherwise
int is_prime(int n) {
    if (n < 2) return 0;
    if (n == 2) return 1;
    if (n % 2 == 0) return 0;
    for (int i = 3; i <= (int)sqrt((double)n); i += 2)
        if (n % i == 0) return 0;
    return 1;
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
        int next_num = 2;          // next candidate to hand out
        int active_slaves = size - 1;
        int msg;
        MPI_Status status;
        printf("Primes up to %d:\n", MAX_VAL);

        while (active_slaves > 0) {
            // Wait for any slave to report
            MPI_Recv(&msg, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG,
                     MPI_COMM_WORLD, &status);
            int slave = status.MPI_SOURCE;

            // Interpret the reply
            if (msg > 0 && msg != 0) {
                // slave just started (msg==0 handled separately) or found a prime
                // Actually: 0 = starting, positive = prime, negative = not prime
            }
            if (msg == 0) {
                ; // slave just started, no result to print
            } else if (msg > 0) {
                printf("  PRIME: %d\n", msg);
            }
            // msg < 0 → not prime, ignore

            // Send next number or a sentinel (-1) to terminate
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

            int result = is_prime(num_to_test) ? num_to_test : -num_to_test;
            MPI_Send(&result, 1, MPI_INT, 0, TAG_WORK, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
