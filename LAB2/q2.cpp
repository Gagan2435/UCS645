#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <omp.h>
#include <algorithm>

using namespace std;

// Scoring
#define MATCH 2
#define MISMATCH -1
#define GAP -1

// DNA Sequences
string seq1 = "ACGTCGATCGATCGATCGATCGATCG";
string seq2 = "ACGTCGATGATCGATCGTTCGATCG";

// Smith-Waterman Function (Wavefront Parallel)
double smith_waterman(int threads) {

    int m = seq1.length();
    int n = seq2.length();

    vector<vector<int>> H(m+1, vector<int>(n+1, 0));

    omp_set_num_threads(threads);

    double start = omp_get_wtime();

    // Wavefront Parallelization
    for (int diag = 2; diag <= m + n; diag++) {

        #pragma omp parallel for schedule(dynamic)
        for (int i = max(1, diag - n); i <= min(m, diag - 1); i++) {

            int j = diag - i;

            if (j >= 1 && j <= n) {

                int score;

                if (seq1[i-1] == seq2[j-1])
                    score = MATCH;
                else
                    score = MISMATCH;

                int diag_score = H[i-1][j-1] + score;
                int up_score   = H[i-1][j] + GAP;
                int left_score = H[i][j-1] + GAP;

                H[i][j] = max(0,
                            max(diag_score,
                            max(up_score, left_score)));
            }
        }
    }

    double end = omp_get_wtime();

    return end - start;
}

int main() {

    ofstream file("q2_data.txt");

    cout << "----------------------------------------\n";
    cout << "Smith-Waterman (OpenMP Wavefront)\n";
    cout << "----------------------------------------\n";
    cout << "Threads   Time(s)\n";
    cout << "----------------------------------------\n";

    for (int t = 1; t <= 4; t++) {

        double time = smith_waterman(t);

        cout << t << "         " << time << endl;

        file << t << " " << time << endl;
    }

    file.close();

    cout << "----------------------------------------\n";
    cout << "Data saved in q2_data.txt\n";

    return 0;
}
