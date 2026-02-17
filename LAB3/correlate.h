#ifndef CORRELATE_H
#define CORRELATE_H

// Q1: Sequential
void correlate_sequential(int ny, int nx, const float* data, float* result);

// Q2: OpenMP Parallel
void correlate_openmp(int ny, int nx, const float* data, float* result);

// Q3: Fully Optimized
void correlate_optimized(int ny, int nx, const float* data, float* result);

#endif
