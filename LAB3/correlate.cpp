#include "correlate.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <omp.h>

// =============================================
// Q1: SEQUENTIAL
// =============================================
void correlate_sequential(int ny, int nx, const float* data, float* result) {
    std::vector<double> norm((long long)ny * nx);

    for (int i = 0; i < ny; i++) {
        double mean = 0.0;
        for (int x = 0; x < nx; x++)
            mean += (double)data[x + i * nx];
        mean /= nx;

        double variance = 0.0;
        for (int x = 0; x < nx; x++) {
            double val = (double)data[x + i * nx] - mean;
            norm[x + (long long)i * nx] = val;
            variance += val * val;
        }
        double stddev = std::sqrt(variance);
        double inv = (stddev < 1e-10) ? 1.0 : 1.0 / stddev;
        for (int x = 0; x < nx; x++)
            norm[x + (long long)i * nx] *= inv;
    }

    for (int i = 0; i < ny; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0.0;
            for (int x = 0; x < nx; x++)
                sum += norm[x + (long long)i * nx] * norm[x + (long long)j * nx];
            result[i + j * ny] = (float)(sum / nx);
        }
    }
}

// =============================================
// Q2: OPENMP PARALLEL
// =============================================
void correlate_openmp(int ny, int nx, const float* data, float* result) {
    std::vector<double> norm((long long)ny * nx);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < ny; i++) {
        double mean = 0.0;
        for (int x = 0; x < nx; x++)
            mean += (double)data[x + i * nx];
        mean /= nx;

        double variance = 0.0;
        for (int x = 0; x < nx; x++) {
            double val = (double)data[x + i * nx] - mean;
            norm[x + (long long)i * nx] = val;
            variance += val * val;
        }
        double stddev = std::sqrt(variance);
        double inv = (stddev < 1e-10) ? 1.0 : 1.0 / stddev;
        for (int x = 0; x < nx; x++)
            norm[x + (long long)i * nx] *= inv;
    }

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0.0;
            for (int x = 0; x < nx; x++)
                sum += norm[x + (long long)i * nx] * norm[x + (long long)j * nx];
            result[i + j * ny] = (float)(sum / nx);
        }
    }
}

// =============================================
// Q3: FULLY OPTIMIZED (SIMD + OpenMP + Cache)
// =============================================
void correlate_optimized(int ny, int nx, const float* data, float* result) {
    std::vector<double> norm((long long)ny * nx);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < ny; i++) {
        double mean = 0.0;
        #pragma omp simd reduction(+:mean)
        for (int x = 0; x < nx; x++)
            mean += (double)data[x + i * nx];
        mean /= nx;

        double variance = 0.0;
        for (int x = 0; x < nx; x++) {
            double val = (double)data[x + i * nx] - mean;
            norm[x + (long long)i * nx] = val;
            variance += val * val;
        }
        double stddev = std::sqrt(variance);
        double inv = (stddev < 1e-10) ? 1.0 : 1.0 / stddev;
        #pragma omp simd
        for (int x = 0; x < nx; x++)
            norm[x + (long long)i * nx] *= inv;
    }

    const int BLOCK = 64;
    #pragma omp parallel for schedule(dynamic, 4)
    for (int ii = 0; ii < ny; ii += BLOCK) {
        for (int jj = 0; jj <= ii; jj += BLOCK) {
            int imax = std::min(ii + BLOCK, ny);
            int jmax = std::min(jj + BLOCK, ii + 1);
            for (int i = ii; i < imax; i++) {
                int jlim = (jj == ii) ? i + 1 : jmax;
                for (int j = jj; j < jlim; j++) {
                    double sum = 0.0;
                    #pragma omp simd reduction(+:sum)
                    for (int x = 0; x < nx; x++)
                        sum += norm[x + (long long)i * nx] * norm[x + (long long)j * nx];
                    result[i + j * ny] = (float)(sum / nx);
                }
            }
        }
    }
}
