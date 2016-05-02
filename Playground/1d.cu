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

#define index_for_A(i, j, n, m, k) ((i * (m + k)) + j)
#define index_for_B(i, j, n, m, k) ((i * (m + k)) + m + j)
#define index_for_X(i, j, n, m, k) ((i * k) + j)

void print_matrix(double *M, int n, int m, int k)
{
    cout.setf(ios::scientific);
    cout.precision(10);
    for (int i = 0; i < m; i++) {
        int index = index_for_X(i, 0, n, m, k);
        cout << M[index];
        for (int j = 1; j < k; j++) {
            index = index_for_X(i, j, n, m, k);
            cout << " " << M[index];
        }
        cout << endl;
    }
}

void swap_rows(int *prm, int i, int j)
{
    int temp = prm[i];
    prm[i] = prm[j];
    prm[j] = temp;
}

void solve_equation(double *M, double *X, int n, int m, int k)
{
    int *prm = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        prm[i] = i;
    }

    int *x_index = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        x_index[i] = i;
    }

    int row = 0;
    int col = 0;

    for ( ; row < n && col < m; row++, col++) {

        int max_value_row = row;
        for (int i = row + 1; i < n; i++) {
            int index_1 = index_for_A(prm[i], col, n, m, k);
            int index_2 = index_for_A(prm[max_value_row], col, n, m, k);
            if (fabs(M[index_1]) > fabs(M[index_2])) {
                max_value_row = i;
            }
        }

        swap_rows(prm, row, max_value_row);

        int index_3 = index_for_A(prm[row], col, n, m, k);
        if (fabs(M[index_3]) > eps) {

            x_index[row] = col;

            for (int i = row + 1; i < n; i++) {

                int num_index = index_for_A(prm[i], col, n, m, k);
                int den_index = index_for_A(prm[row], col, n, m, k);

                double factor = -M[num_index] / M[den_index];
                for (int j = col; j < (m + k); j++) {
                    int left_index = index_for_A(prm[i], j, n, m, k);
                    int right_index = index_for_A(prm[row], j, n, m, k);
                    M[left_index] += M[right_index] * factor;
                }
                // for (int j = 0; j < k; j++) {
                //     B[prm[i]][j] += B[prm[row]][j] * factor;
                // }
            }
        } else {
            row--;
        }

    }

    if (row == n || col == m) { // ??
        row--;
    }

    for (int t = 0; t < k; t++) {
        for (int i = row; i >= 0; i--) {
            int index = x_index[i];
            double sum = 0.0;
            for (int j = index + 1; j < m; j++) {
                int A_index = index_for_A(prm[i], j, n, m, k);
                int X_index = index_for_X(j, t, n, m, k);
                sum += M[A_index] * X[X_index];
            }
            int pivot_index = index_for_A(prm[i], index, n, m, k);
            int target_X_index = index_for_X(index, t, n, m, k);
            if (fabs(M[pivot_index]) > eps) {
                int B_index = index_for_B(prm[i], t, n, m, k);
                X[target_X_index] = (M[B_index] - sum) / M[pivot_index];
            } else {
                X[target_X_index] = 0.0;
            }
        }
    }
}

int main()
{
    int n, m, k;
    cin >> n >> m >> k;

    int M_size = (n * m) + (n * k);

    double *M = (double *)malloc(M_size * sizeof(double));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            int index = index_for_A(i, j, n, m, k);
            cin >> M[index];
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            int index = index_for_B(i, j, n, m, k);
            cin >> M[index];
        }
    }

    int X_size = m * k;

    double *X = (double *)malloc(X_size * sizeof(double));

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            int index = index_for_X(i, j, n, m, k);
            X[index] = 0.0;
        }
    }

    solve_equation(M, X, n, m, k);
    print_matrix(X, n, m, k);

    return 0;
}
