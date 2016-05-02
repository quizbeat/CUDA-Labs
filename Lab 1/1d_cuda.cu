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

#define CSC(call) {														\
    cudaError err = call;												\
    if(err != cudaSuccess) {											\
        fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",	\
            __FILE__, __LINE__, cudaGetErrorString(err));				\
        exit(1);														\
    }																	\
} while (0)

using namespace std;

const double eps = 10e-7;

#define index_for_A(i, j, n, m, k) ((i * (m + k)) + j)
#define index_for_B(i, j, n, m, k) ((i * (m + k)) + m + j)
#define index_for_X(i, j, n, m, k) ((i * k) + j)

void print_matrix(double *M, int n, int m, int k) {
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

__global__ void initSequence(int *seq, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;
    for ( ; index < n; index += offset) {
        seq[index] = index;
    }
}

__global__ void findMaxRowAndSwap(double *M, int *prm, int row, int col, int n, int m, int k) {
    int max_value_row = row;
    for (int i = row + 1; i < n; i++) {
        int index_current = index_for_A(prm[i], col, n, m, k);
        int index_max = index_for_A(prm[max_value_row], col, n, m, k);
        if (fabs(M[index_current]) > fabs(M[index_max])) {
            max_value_row = i;
        }
    }
    if (max_value_row != row) {
        int temp = prm[row];
        prm[row] = prm[max_value_row];
        prm[max_value_row] = temp;
    }
}

__global__ void updateRows(double *M, int *prm, int *pivot_for_row, int row, int col, int n, int m, int k) {
    pivot_for_row[row] = col;

    for (int i = row + 1; i < n; i++) {
        int factor_numerator_index = index_for_A(prm[i], col, n, m, k);
        int factor_denominator_index = index_for_A(prm[row], col, n, m, k);
        double factor = -M[factor_numerator_index] / M[factor_denominator_index];

        int j = col + blockIdx.x * blockDim.x + threadIdx.x;
        int offset = gridDim.x * blockDim.x;

        for ( ; j < (m + k); j += offset) {
            int target_item_index = index_for_A(prm[i], j, n, m, k);
            int pivot_item_index = index_for_A(prm[row], j, n, m, k);
            M[target_item_index] += M[pivot_item_index] * factor;
        }
    }
}

__global__ void computeX(double *M, double *X, int *prm, int *pivot_for_row, int row, int n, int m, int k) {
    for (int t = 0; t < k; t++) {

        for (int i = row; i >= 0; i--) {

            int index = pivot_for_row[i];
            double sum = 0.0;

            for (int j = index + 1; j < m; j++) {
                int A_index = index_for_A(prm[i], j, n, m, k);
                int X_index = index_for_X(j, t, n, m, k);
                sum += M[A_index] * X[X_index];
            }
            int A_index = index_for_A(prm[i], index, n, m, k);
            int X_target_index = index_for_X(index, t, n, m, k);

            if (fabs(M[A_index]) > eps) {
                int B_index = index_for_B(prm[i], t, n, m, k);
                X[X_target_index] = (M[B_index] - sum) / M[A_index];
            } else {
                X[X_target_index] = 0.0;
            }
        }
    }
}

int main()
{
    int n, m, k;
    cin >> n >> m >> k;

    int M_size = (n * m) + (n * k);
    double *M_host = (double *)malloc(M_size * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            int index = index_for_A(i, j, n, m, k);
            cin >> M_host[index];
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            int index = index_for_B(i, j, n, m, k);
            cin >> M_host[index];
        }
    }
    double *M_device = NULL;
    cudaMalloc((void **)&M_device, M_size * sizeof(double));
    CSC(cudaGetLastError());
    cudaMemcpy(M_device, M_host, M_size * sizeof(double), cudaMemcpyHostToDevice);
    CSC(cudaGetLastError());

    int X_size = m * k;
    double *X_host = (double *)malloc(X_size * sizeof(double));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            int index = index_for_X(i, j, n, m, k);
            X_host[index] = 0.0;
        }
    }
    double *X_device = NULL;
    cudaMalloc((void **)&X_device, X_size * sizeof(double));
    CSC(cudaGetLastError());
    cudaMemcpy(X_device, X_host, X_size * sizeof(double), cudaMemcpyHostToDevice);
    CSC(cudaGetLastError());

    int *prm = NULL;
    cudaMalloc((void **)&prm, n * sizeof(int));
    CSC(cudaGetLastError());
    initSequence <<<32, 32>>> (prm, n);
    CSC(cudaGetLastError());

    int *pivot_for_row = NULL;
    cudaMalloc((void **)&pivot_for_row, n * sizeof(int));
    CSC(cudaGetLastError());
    initSequence <<<32, 32>>> (pivot_for_row, n);
    CSC(cudaGetLastError());



    int row = 0;
    int col = 0;

    for ( ; row < n && col < m; row++, col++) {

        findMaxRowAndSwap <<<1, 1>>> (M_device, prm, row, col, n, m, k);
        CSC(cudaGetLastError());
        cudaThreadSynchronize();

        int prm_value;
        cudaMemcpy(&prm_value, &prm[row], sizeof(int), cudaMemcpyDeviceToHost);
        CSC(cudaGetLastError());
        cudaThreadSynchronize();

        double pivot_value = 0;
        int pivot_index = index_for_A(prm_value, col, n, m, k);
        cudaMemcpy(&pivot_value, &M_device[pivot_index], sizeof(double), cudaMemcpyDeviceToHost);
        CSC(cudaGetLastError());
        cudaThreadSynchronize();

        if (fabs(pivot_value) > eps) {
            updateRows <<<32, 32>>> (M_device, prm, pivot_for_row, row, col, n, m, k);
            CSC(cudaGetLastError());
            cudaThreadSynchronize();
        } else {
            row--;
        }
    }

    if (row == n || col == m) {
        row--;
    }

    computeX <<<1, 1>>> (M_device, X_device, prm, pivot_for_row, row, n, m, k);
    CSC(cudaGetLastError());
    cudaThreadSynchronize();

    cudaMemcpy(X_host, X_device, X_size * sizeof(double), cudaMemcpyDeviceToHost);
    CSC(cudaGetLastError());
    cudaThreadSynchronize();

    print_matrix(X_host, n, m, k);

    cudaFree(M_device);
    cudaFree(X_device);
    cudaFree(prm);
    cudaFree(pivot_for_row);

    free(M_host);
    free(X_host);

    return 0;
}
