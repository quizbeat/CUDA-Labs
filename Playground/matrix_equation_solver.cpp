//
//  main.cpp
//  matrix-test
//
//  Created by Nikita Makarov on 22/03/16.
//  Copyright © 2016 Nikita Makarov. All rights reserved.
//

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iomanip>

using namespace std;

void print_matrix(float **M, long n, long m)
{
    cout << fixed;
    for (long i = 0; i < n; i++) {
        for (long j = 0; j < m; j++) {
            cout << setprecision(10) << M[i][j] << '\t';
        }
        cout << endl;
    }
}

void print_matrix_with_prm(float **M, long *prm, long n, long m)
{
    for (long i = 0; i < n; i++) {
        for (long j = 0; j < m; j++) {
            cout << M[prm[i]][j] << " ";
        }
        cout << endl;
    }
}

void swap_rows(long *prm, long i, long j)
{
    long temp = prm[i];
    prm[i] = prm[j];
    prm[j] = temp;
}

bool zero_column(float **A, long *prm, long n, long i, long j)
// returns true if column j in the rows from i to n equals zero
{
    for ( ; i < n; i++) {
        if (A[prm[i]][j] != 0) {
            return false;
        }
    }
    return true;
}

void solve_equation(float **A, float **X, float **B, long n, long m, long k)
// solves matrix equation AX = B, where:
// A - n x m matrix, B - n x k matrix, X - unknown m x k matrix
{
    // create permutations array
    long *prm = (long *)malloc(n * sizeof(long));
    for (long i = 0; i < n; i++) {
        prm[i] = i;
    }

    long *x_index = (long *)malloc(n * sizeof(long));
    x_index[0] = 0;

    // transform block matrix [A|B] to the row echelon form
    long j_pivot = 0;
    for (long row = 0; row < n; row++) {
        long column = row;
        // find row with max value at current column
        long max_value_row = row;
        for (long i = row + 1; i < n; i++) {
            if (fabs(A[prm[i]][column] > A[prm[max_value_row]][column])) {
                max_value_row = i;
            }
        }

        // swap current row and found row
        swap_rows(prm, row, max_value_row);

        // update bottom rows
        for (long i = row + 1; i < n; i++) {
            float factor = -A[prm[i]][j_pivot] / A[prm[row]][j_pivot];
            for (long j = j_pivot; j < m; j++) {
                A[prm[i]][j] += A[prm[row]][j] * factor;
            }
            for (long j = 0; j < k; j++) {
                B[prm[i]][j] += B[prm[row]][j] * factor;
            }
        }

        // check next columns for zero elements
        for (long j = j_pivot; j < m; j++) {
            if (!zero_column(A, prm, n, row + 1, j)) {
                x_index[row + 1] = j; // mark for future search of x index
                break;
            }
            j_pivot = j - 1; // - 1 because it will increments for next iteration
        }
        j_pivot++;
    }

    // calculate matrix X
    // for each column from X
    for (long t = 0; t < k; t++) {
        // for each row in [A|B] from end to begin
        for (long i = (n - 1); i >= 0; i--) {
            long index = x_index[i];
            float sum = 0.0;
            for (long j = index + 1; j < m; j++) {
                sum += A[prm[i]][j] * X[j][t];
            }
            X[index][t] = (B[prm[i]][t] - sum) / A[prm[i]][index];
        }
    }
}

int main()
{
    long n, m, k;
    cin >> n >> m >> k;

    // allocating memory for matrix A
    float **A = (float **)malloc(n * sizeof(float *));
    for (long i = 0; i < n; i++) {
        A[i] = (float *)malloc(m * sizeof(float));
    }

    // reading matrix A
    for (long i = 0; i < n; i++) {
        for (long j = 0; j < m; j++) {
            cin >> A[i][j];
        }
    }

    // allocating memory for matrix B
    float **B = (float **)malloc(n * sizeof(float *));
    for (long i = 0; i < n; i++) {
        B[i] = (float *)malloc(k * sizeof(float));
    }

    // reading matrix B
    for (long i = 0; i < n; i++) {
        for (long j = 0; j < k; j++) {
            cin >> B[i][j];
        }
    }

    // allocating memory for matrix X
    float **X = (float **)malloc(m * sizeof(float *));
    for (long i = 0; i < m; i++) {
        X[i] = (float *)malloc(k * sizeof(float));
    }

    // init matrix X
    for (long i = 0; i < m; i++) {
        for (long j = 0; j < k; j++) {
            X[i][j] = 0.0;
        }
    }

    solve_equation(A, X, B, n, m, k);
    print_matrix(X, m, k);

    return 0;
}
