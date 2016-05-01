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

#define CSC(call) {														\
    cudaError err = call;												\
    if(err != cudaSuccess) {											\
        fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",	\
            __FILE__, __LINE__, cudaGetErrorString(err));				\
        exit(1);														\
    }																	\
} while (0)

const double eps = 10e-7;

__host__ void print_matrix(double **M, int n, int m) {
    cout.setf(ios::scientific);
    cout.precision(10);
    for (int i = 0; i < n; i++) {
        cout << M[i][0];
        for (int j = 1; j < m; j++) {
            cout << " " << M[i][j];
        }
        cout << endl;
    }
}

__global__ void swap_rows(int *prm, int i, int j) {
    int temp = prm[i];
    prm[i] = prm[j];
    prm[j] = temp;
}

// inits given array with numbers from 0 to n - 1
__global__ void initSequence(double *seq, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;
    for ( ; index < n; index += offset) {
        seq[index] = index;
    }
}

// updates rows from row_start to n, columns from col_start to (m + k)
// concurrently updates all columns and all rows ??
__global__ void updateRowsBelow(double *M, int row_start, int col_start, int n, int m, int k) {
    int row_index = 0;

    for (int i = row + 1; i < n; i++) {

        double factor = -A[prm[i]][col] / A[prm[row]][col];
        for (int j = col; j < m; j++) {
            A[prm[i]][j] += A[prm[row]][j] * factor;
        }
        for (int j = 0; j < k; j++) {
            B[prm[i]][j] += B[prm[row]][j] * factor;
        }
    }
}

__global__ void backSubstitution(double *M, double *X, double *x_index, int row, int n, int m, int k) {
    int X_column_index = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;

    for ( ; X_column_index < k; X_column_index += offset) {

        for (int i = row; i >= 0; i--) {

            int index = x_index[i];
            double sum = 0.0;

            for (int j = index + 1; j < m; j++) {
                int A_index = index_for_A(prm[i], j, n, m, k);
                int X_index = index_for_X(j, X_column_index, n, m, k);
                sum += M[A_index] * X[X_index];
            }

            int A_index = index_for_A(prm[i], index, n, m, k);
            int X_target_index = index_for_X(index, X_column_index, n, m, k);

            if (fabs(A[prm[i]][index]) > eps) {
                int B_index = index_for_B(prm[i], X_column_index, n, m, k);
                X[X_target_index] = (M[B_index] - sum) / M[A_index];
            } else {
                X[X_target_index] = 0.0;
            }
        }
    }
    __syncthreads();
}

void solve_equation(double **A, double **X, double **B, int n, int m, int k)
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
            if (fabs(A[prm[i]][col]) > fabs(A[prm[max_value_row]][col])) {
                max_value_row = i;
            }
        }

        swap_rows(prm, row, max_value_row);

        if (fabs(A[prm[row]][col]) > eps) {

            x_index[row] = col;

            for (int i = row + 1; i < n; i++) {

                double factor = -A[prm[i]][col] / A[prm[row]][col];
                for (int j = col; j < m; j++) {
                    A[prm[i]][j] += A[prm[row]][j] * factor;
                }
                for (int j = 0; j < k; j++) {
                    B[prm[i]][j] += B[prm[row]][j] * factor;
                }
            }
        } else {
            row--;
        }

    }

    if (row == n || col == m) {
        row--;
    }

    for (int t = 0; t < k; t++) {
        for (int i = row; i >= 0; i--) {
            int index = x_index[i];
            double sum = 0.0;
            for (int j = index + 1; j < m; j++) {
                sum += A[prm[i]][j] * X[j][t];
            }
            if (fabs(A[prm[i]][index]) > eps) {
                X[index][t] = (B[prm[i]][t] - sum) / A[prm[i]][index];
            } else {
                X[index][t] = 0.0;
            }
        }
    }
}

__host__ inline int index_for_A(int i, int j, int n, int m, int k) {
    return (i * (m + k)) + j;
}

__host__ inline int index_for_B(int i, int j, int n, int m, int k) {
    return (i * (m + k)) + m + j;
}

__host__ inline int index_for_X(int i, int j, int n, int m, int k) {
    return i *
}

int main()
{
    int n, m, k;
    cin >> n >> m >> k;

    // Solving next equation: A * X = B
    // Let's M = [A|B]
    // M = [{A_row_1}{B_row_1},...,{A_row_n},{B_row_n}]

    int M_size = (n * m) + (n * k);

    double *M_host = (double *)malloc(M_size * sizeof(double));

    // read matrix A
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            int index = index_for_A(i, j, n, m, k);
            cin >> M_host[index];
        }
    }

    // read matrix B
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            int index = index_for_B(i, j, n, m, k);
            cin >> M_host[index];
        }
    }

    // X = [{X_row_1},...,{X_row_m}]

    int X_size = m * k;

    double *X_host = (double *)malloc(X_size * sizeof(double));

    // init matrix X
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            int index = index_for_X(i, j, n, m, k);
            X_host[index] = 0.0;
        }
    }

    // Begin solving equation

    double *M_device; // matrix M on device
    double *X_device; // matrix X on device

    // alloc memory on device for matrix M
    CSC(cudaMalloc((void **)&M_device, M_size * sizeof(double)));
    // CSC(cudaMalloc((void **)&X_device, X_size * sizeof(double)));

    // copy matrix data
    CSC(cudaMemcpy(M_device, M_host, M_size * sizeof(double), cudaMemcpyHostToDevice));

    int *prm_host;   // rows permutations array on host
    int *prm_device; // rows permutations array on device

    // alloc memory for permutations array
    prm_host = (int *)malloc(n * sizeof(int));
    CSC(cudaMalloc((void **)&prm_device, n * sizeof(int)));

    // init permutations array on device
    initSequence <<<32, 32>>> (prm_device, n);

    int *x_index_host;   // array with indexes for diagonal elements on host
    int *x_index_device; // array with indexes for diagonal elements on device

    // alloc memory for array with indexes
    x_index_host = (int *)malloc(n * sizeof(int));
    CSC(cudaMalloc((void **)&x_index_device, n * sizeof(int)));

    // init indexes on device
    initSequence <<<32, 32>>> (x_index_device, n);

    int row = 0;
    int col = 0;

    for ( ; row < n && col < m; row++, col++) {

        // find row with max value
        int max_value_row = row;
        for (int i = row + 1; i < n; i++) {
            int index_current = index_for_A(prm_host[i], col, n, m, k);
            int index_max = index_for_A(prm[max_value_row], col, n, m, k);
            if (fabs(M_host[index_current]) > fabs(M_host[index_max])) {
                max_value_row = i;
            }
        }

        // swap rows on device
        swap_rows <<<1, 1>>> (prm_device, row, max_value_row);

        // copy permutations array from device to host
        CSC(cudaMemcpy(prm_host, prm_device, n * sizeof(int), cudaMemcpyDeviceToHost));

        // index for current pivot element
        int index = index_for_A(prm_host[row], col, n, m, k);

        double *M_pivot_element = (double *)malloc(sizeof(double)); // ???

        // copy pivot value from device to host
        CSC(cudaMemcpy(M_pivot_element, M_device[index], sizeof(double), cudaMemcpyDeviceToHost));

        if (fabs(*M_pivot_element) > eps) {



            for ()

            // update rows below
            updateRowsBelow <<<32, 32>>> (M_device, row, col, n, m, k);

            // remember position of diagonal element
            x_index[row] = col;

            // copy updated x_index array to device
            CSC(cudaMemcpy(x_index_device, x_index_host, n * sizeof(int), cudaMemcpyHostToDevice));


        } else {
            // need to perform next iter on current row
            row--;
        }

    }

    if (row == n || col == m) {
        row--;
    }

    for (int t = 0; t < k; t++) {
        for (int i = row; i >= 0; i--) {
            int index = x_index[i];
            double sum = 0.0;
            for (int j = index + 1; j < m; j++) {
                sum += A[prm[i]][j] * X[j][t];
            }
            if (fabs(A[prm[i]][index]) > eps) {
                X[index][t] = (B[prm[i]][t] - sum) / A[prm[i]][index];
            } else {
                X[index][t] = 0.0;
            }
        }
    }



    return 0;
}
