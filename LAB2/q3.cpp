#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

// Grid size
#define N 200

// Time steps
#define STEPS 500

// Diffusion constant
#define ALPHA 0.25


// Function to run simulation
double run_simulation(int threads) {

    static double temp[N][N];
    static double new_temp[N][N];

    // Initialize temperature
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){

            if(i==0 || j==0 || i==N-1 || j==N-1)
                temp[i][j] = 100.0;   // Hot boundary
            else
                temp[i][j] = 0.0;     // Cold inside
        }
    }

    omp_set_num_threads(threads);

    double start = omp_get_wtime();

    // Time iterations
    for(int t=0;t<STEPS;t++){

        double total_heat = 0.0;

        // Parallel spatial loop
        #pragma omp parallel for reduction(+:total_heat) schedule(static)
        for(int i=1;i<N-1;i++){
            for(int j=1;j<N-1;j++){

                new_temp[i][j] =
                    temp[i][j] +
                    ALPHA * (
                    temp[i+1][j] +
                    temp[i-1][j] +
                    temp[i][j+1] +
                    temp[i][j-1] -
                    4*temp[i][j]);

                total_heat += new_temp[i][j];
            }
        }

        // Copy new -> old
        #pragma omp parallel for schedule(static)
        for(int i=1;i<N-1;i++){
            for(int j=1;j<N-1;j++){
                temp[i][j] = new_temp[i][j];
            }
        }
    }

    double end = omp_get_wtime();

    return end - start;
}


// Main function
int main(){

    cout << "----------------------------------------" << endl;
    cout << "Heat Diffusion Simulation (OpenMP)" << endl;
    cout << "Grid Size : " << N << " x " << N << endl;
    cout << "Steps     : " << STEPS << endl;
    cout << "----------------------------------------" << endl;
    cout << "Threads\tTime(s)" << endl;
    cout << "----------------------------------------" << endl;

    // Run for 1 to 4 threads
    for(int t=1;t<=4;t++){

        double time = run_simulation(t);

        cout << t << "\t" << time << endl;
    }

    cout << "----------------------------------------" << endl;

    return 0;
}
