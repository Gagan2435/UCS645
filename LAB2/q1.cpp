#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

using namespace std;

// Number of particles
#define N 1000

// Lennard-Jones parameters
const double epsilon = 1.0;
const double sigma = 1.0;
const double cutoff = 2.5 * sigma;

int main() {

    // Position arrays
    vector<double> x(N), y(N), z(N);

    // Force arrays
    vector<double> fx(N, 0.0), fy(N, 0.0), fz(N, 0.0);

    // Initialize random positions
    for (int i = 0; i < N; i++) {
        x[i] = drand48() * 10.0;
        y[i] = drand48() * 10.0;
        z[i] = drand48() * 10.0;
    }

    double total_energy = 0.0;

    // Start timer
    double start = omp_get_wtime();

    // Parallel region
    #pragma omp parallel
    {
        // Private force arrays (avoid race conditions)
        vector<double> fx_local(N, 0.0);
        vector<double> fy_local(N, 0.0);
        vector<double> fz_local(N, 0.0);

        double energy_local = 0.0;

        // Parallel nested loops
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < N; i++) {

            for (int j = i + 1; j < N; j++) {

                double dx = x[i] - x[j];
                double dy = y[i] - y[j];
                double dz = z[i] - z[j];

                double r2 = dx*dx + dy*dy + dz*dz;
                double r  = sqrt(r2);

                if (r < cutoff) {

                    double sr = sigma / r;
                    double sr6 = pow(sr, 6);
                    double sr12 = sr6 * sr6;

                    // Lennard-Jones potential
                    double potential =
                        4.0 * epsilon * (sr12 - sr6);

                    energy_local += potential;

                    // Force magnitude
                    double force =
                        24.0 * epsilon *
                        (2.0 * sr12 - sr6) / r2;

                    double fx_ij = force * dx;
                    double fy_ij = force * dy;
                    double fz_ij = force * dz;

                    // Local accumulation
                    fx_local[i] += fx_ij;
                    fy_local[i] += fy_ij;
                    fz_local[i] += fz_ij;

                    fx_local[j] -= fx_ij;
                    fy_local[j] -= fy_ij;
                    fz_local[j] -= fz_ij;
                }
            }
        }

        // Merge local results
        #pragma omp critical
        {
            total_energy += energy_local;

            for (int i = 0; i < N; i++) {
                fx[i] += fx_local[i];
                fy[i] += fy_local[i];
                fz[i] += fz_local[i];
            }
        }
    }

    // End timer
    double end = omp_get_wtime();

    // Output
    cout << "----------------------------------" << endl;
    cout << "Molecular Dynamics (OpenMP)" << endl;
    cout << "Particles      : " << N << endl;
    cout << "Total Energy   : " << total_energy << endl;
    cout << "Execution Time : " << end - start << " seconds" << endl;
    cout << "----------------------------------" << endl;

    return 0;
}
