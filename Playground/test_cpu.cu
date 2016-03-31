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

const double eps = 10e-7;

void print_matrix(double **M, long n, long m)
{
    cout.setf(ios::scientific);
    cout.precision(10);
    for (long i = 0; i < n; i++) {
        cout << M[i][0];
        for (long j = 1; j < m; j++) {
            cout << " " << M[i][j];
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

bool zero_column(double **A, long *prm, long n, long i, long j)
{
    for ( ; i < n; i++) {
        if (A[prm[i]][j] > eps) {
            return false;
        }
    }
    return true;
}

void solve_equation(double **A, double **X, double **B, long n, long m, long k)
{
    long *prm = (long *)malloc(n * sizeof(long));
    for (long i = 0; i < n; i++) {
        prm[i] = i;
    }

    long *x_index = (long *)malloc(n * sizeof(long));
    x_index[0] = 0;

    long j_pivot = 0;
    for (long row = 0; row < n; row++) {
        long column = row;
        long max_value_row = row;
        for (long i = row + 1; i < n; i++) {
            if (fabs(A[prm[i]][column]) > fabs(A[prm[max_value_row]][column])) {
                max_value_row = i;
            }
        }

        swap_rows(prm, row, max_value_row);

        for (long i = row + 1; i < n; i++) {
            double factor = -A[prm[i]][j_pivot] / A[prm[row]][j_pivot];
            for (long j = j_pivot; j < m; j++) {
                A[prm[i]][j] += A[prm[row]][j] * factor;
            }
            for (long j = 0; j < k; j++) {
                B[prm[i]][j] += B[prm[row]][j] * factor;
            }
        }

        for (long j = j_pivot; j < m; j++) {
            if (!zero_column(A, prm, n, row + 1, j)) {
                x_index[row + 1] = j;
                break;
            }
            j_pivot = j - 1;
        }
        j_pivot++;
    }

    for (long t = 0; t < k; t++) {
        for (long i = (n - 1); i >= 0; i--) {
            long index = x_index[i];
            double sum = 0.0;
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

    double **A = (double **)malloc(n * sizeof(double *));
    for (long i = 0; i < n; i++) {
        A[i] = (double *)malloc(m * sizeof(double));
    }

    for (long i = 0; i < n; i++) {
        for (long j = 0; j < m; j++) {
            cin >> A[i][j];
        }
    }

    double **B = (double **)malloc(n * sizeof(double *));
    for (long i = 0; i < n; i++) {
        B[i] = (double *)malloc(k * sizeof(double));
    }

    for (long i = 0; i < n; i++) {
        for (long j = 0; j < k; j++) {
            cin >> B[i][j];
        }
    }

    double **X = (double **)malloc(m * sizeof(double *));
    for (long i = 0; i < m; i++) {
        X[i] = (double *)malloc(k * sizeof(double));
    }

    for (long i = 0; i < m; i++) {
        for (long j = 0; j < k; j++) {
            X[i][j] = 0.0;
        }
    }

    solve_equation(A, X, B, n, m, k);
    print_matrix(X, m, k);

    return 0;
}
