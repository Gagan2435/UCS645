#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <omp.h>
#include "correlate.h"

int main(int argc, char* argv[]) {

    if (argc != 3) {
        std::cerr << "Usage: ./correlate <ny> <nx>\n";
        std::cerr << "Example: ./correlate 500 1000\n";
        return 1;
    }

    int ny = std::atoi(argv[1]);
    int nx = std::atoi(argv[2]);
    int threads = omp_get_max_threads();

    std::cout << "\n==========================================\n";
    std::cout << "   UCS645: Parallel & Distributed Computing\n";
    std::cout << "   Correlation Assignment\n";
    std::cout << "==========================================\n";
    std::cout << "Matrix  : " << ny << " rows x " << nx << " cols\n";
    std::cout << "Threads : " << threads << "\n";
    std::cout << "==========================================\n\n";

    // Generate random input matrix
    std::vector<float> data((long long)ny * nx);
    srand(42);
    for (long long i = 0; i < (long long)ny * nx; i++)
        data[i] = static_cast<float>(rand()) / RAND_MAX;

    std::vector<float> result((long long)ny * ny, 0.0f);

    // Q1: Sequential
    std::cout << "[Q1] Sequential running...\n";
    auto t1 = std::chrono::high_resolution_clock::now();
    correlate_sequential(ny, nx, data.data(), result.data());
    auto t2 = std::chrono::high_resolution_clock::now();
    double seq_time = std::chrono::duration<double>(t2 - t1).count();
    std::cout << "     Time: " << std::fixed << std::setprecision(4) << seq_time << " seconds\n\n";

    // Q2: OpenMP
    std::cout << "[Q2] OpenMP Parallel running (" << threads << " threads)...\n";
    t1 = std::chrono::high_resolution_clock::now();
    correlate_openmp(ny, nx, data.data(), result.data());
    t2 = std::chrono::high_resolution_clock::now();
    double omp_time = std::chrono::duration<double>(t2 - t1).count();
    std::cout << "     Time: " << std::fixed << std::setprecision(4) << omp_time << " seconds\n\n";

    // Q3: Optimized
    std::cout << "[Q3] Fully Optimized running...\n";
    t1 = std::chrono::high_resolution_clock::now();
    correlate_optimized(ny, nx, data.data(), result.data());
    t2 = std::chrono::high_resolution_clock::now();
    double opt_time = std::chrono::duration<double>(t2 - t1).count();
    std::cout << "     Time: " << std::fixed << std::setprecision(4) << opt_time << " seconds\n\n";

    // Speedup Summary
    std::cout << "==========================================\n";
    std::cout << "   SPEEDUP SUMMARY\n";
    std::cout << "==========================================\n";
    std::cout << "Sequential : " << seq_time << "s\n";
    std::cout << "OpenMP     : " << omp_time << "s  (speedup: " << std::setprecision(2) << seq_time/omp_time << "x)\n";
    std::cout << "Optimized  : " << opt_time << "s  (speedup: " << seq_time/opt_time << "x)\n";
    std::cout << "==========================================\n\n";

    // Save to CSV for graphs (Q4)
    std::ofstream csv("results.csv", std::ios::app);
    if (csv.tellp() == 0)
        csv << "ny,nx,threads,seq_time,omp_time,opt_time,speedup_omp,speedup_opt\n";
    csv << ny << "," << nx << "," << threads << ","
        << seq_time << "," << omp_time << "," << opt_time << ","
        << seq_time/omp_time << "," << seq_time/opt_time << "\n";
    csv.close();
    std::cout << "Results saved to results.csv\n";

    return 0;
}
