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

#define index_for_A(i, j, n, m, k) ((i * (m + k)) + j)
#define index_for_B(i, j, n, m, k) ((i * (m + k)) + m + j)
#define index_for_X(i, j, n, m, k) ((i * k) + j)

__host__ void print_matrix(double *M, int n, int m, int k) {
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

// j is a pointer to max value row on device
__global__ void swap_rows(int *prm, int i, int *j) {
    int temp = prm[i];
    prm[i] = prm[*j];
    prm[*j] = temp;
}

// inits given array with numbers from 0 to n - 1
__global__ void initSequence(int *seq, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;
    for ( ; index < n; index += offset) {
        seq[index] = index;
    }
}

// finds max value row beginnig from row to n, updates max_value_row pointer
__global__ void findMaxValueRow(double *M, int *prm, int row, int col, int n, int m, int k, int *max_value_row) {
    *max_value_row = row;
    for (int i = row + 1; i < n; i++) {
        int index_current = index_for_A(prm[i], col, n, m, k);
        int index_max = index_for_A(prm[*max_value_row], col, n, m, k);
        if (fabs(M[index_current]) > fabs(M[index_max])) {
            *max_value_row = i;
        }
    }
}

// updates rows from row_start to n, columns from col_start to (m + k)
// concurrently updates all columns ??
__global__ void updateRowsBelow(double *M, int *prm, int *x_index, int row, int col, int n, int m, int k) {

    for (int i = row + 1; i < n; i++) {

        int factor_numerator_index = index_for_A(prm[i], col, n, m, k);
        int factor_denominator_index = index_for_A(prm[row], col, n, m, k);  // reusing every iter ??
        double factor = -M[factor_numerator_index] / M[factor_denominator_index];

        int column_index = col + blockIdx.x * blockDim.x + threadIdx.x; /// !!!!!!
        int offset = gridDim.x * blockDim.x;

        for ( ; column_index < (m + k); column_index += offset) {
            int target_item_index = index_for_A(prm[i], column_index, n, m, k);
            int pivot_item_index = index_for_A(prm[row], column_index, n, m, k);
            M[target_item_index] += M[pivot_item_index] * factor;
        }
    }

#ifdef DEBUG
    printf("----------- setting x_index[%d] = %d\n", row, col);
#endif

    x_index[row] = col;
}

__global__ void backSubstitution(double *M, double *X, int *prm, int *x_index, int row, int n, int m, int k) {
    int X_column_index = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;

#ifdef DEBUG
    // printf(">>> backSubstitution: X_column_index = %d\n", X_column_index);
    // printf(">>> backSubstitution: offset = %d\n", offset);
    // printf(">>> backSubstitution: initial row = %d\n", row);

    printf("\n>>>>> permutations array: \n");
    for (int p = 0; p < n; p++) {
        printf("%d ", prm[p]);
    }
    printf("\n\n");

    printf("\n>>>>> x_index array: \n");
    for (int p = 0; p < n; p++) {
        printf("%d ", x_index[p]);
    }
    printf("\n\n");

#endif

    for ( ; X_column_index < k; X_column_index += offset) {

#ifdef DEBUG
    printf("\n\n>>> backSubstitution for X column [%d]\n", X_column_index);
#endif

        for (int i = row; i >= 0; i--) {

            int index = x_index[i];
            double sum = 0.0;

#ifdef DEBUG
            printf("\n\n\n     Calculating sum of known x values on row [%d]\n", i);
            printf("       Sum: ");
#endif

            for (int j = index + 1; j < m; j++) {
                int A_index = index_for_A(prm[i], j, n, m, k);
                int X_index = index_for_X(j, X_column_index, n, m, k);
                sum += M[A_index] * X[X_index];
#ifdef DEBUG
                printf("(%f * %f)", M[A_index], X[X_index]);
                if (j != m - 1) {
                    printf(" + ");
                }
#endif
            }

#ifdef DEBUG
            printf("\n      Sum = %f\n", sum);
#endif

            int A_index = index_for_A(prm[i], index, n, m, k);
            int X_target_index = index_for_X(index, X_column_index, n, m, k);

#ifdef DEBUG
            printf("     Current pivot value = %f\n", M[A_index]);
            printf("     Changing X element with indexes (%d, %d)\n", index, X_column_index);
#endif

            if (fabs(M[A_index]) > eps) {
                int B_index = index_for_B(prm[i], X_column_index, n, m, k);
                X[X_target_index] = (M[B_index] - sum) / M[A_index];
#ifdef DEBUG
                printf("    X item non zero, calculated from  [ (%f - %f) / %f ] = [%f]\n", M[B_index], sum, M[A_index], X[X_target_index]);
#endif
            } else {
#ifdef DEBUG
                printf("      X is zero\n");
#endif
                X[X_target_index] = 0.0;
            }

#ifdef DEBUG
            printf("Calculating column [%d] with row [%d]\n", X_column_index, i);
            for (int q = 0; q < m; q++) {
                int index = index_for_X(q, 0, n, m, k);
                printf("%f ", X[index]);
                for (int p = 1; p < k; p++) {
                    index = index_for_X(q, p, n, m, k);
                    printf(" %f", X[index]);
                }
                printf("\n");
            }
            printf("\n");
#endif
        }
    }
}

__global__ void printMatrix(double *M, int *prm, int n, int m, int k) {
    printf("------------------------ MATRIX M ---------------------------\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < (m + k); j++) {
            if (j == m) {
                printf("| ");
            }
            int index = index_for_A(prm[i], j, n, m, k);
            printf("%lf ", M[index]);
        }
        printf("\n");
    }
    printf("---------------------------------------------------------------\n\n");
}

int main() {

    int n, m, k;
    cin >> n >> m >> k;

#ifdef DEBUG
    cout << "Matrix A: " << n << " row(s), " << m << " column(s)\n";
    cout << "Matrix B: " << n << " row(s), " << k << " column(s)\n";
    cout << "Matrix X: " << m << " row(s), " << k << " column(s)\n";
#endif

    // Solving next equation: A * X = B
    // Let's M = [A|B]
    // M = [{A_row_1}{B_row_1},...,{A_row_n},{B_row_n}]

    int M_size = (n * m) + (n * k);

#ifdef DEBUG
    cout << "> alloc memory for M_host\n";
#endif

    double *M_host = (double *)malloc(M_size * sizeof(double));

    // read matrix A

#ifdef DEBUG
    cout << "> read M_host part A\n";
#endif

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            int index = index_for_A(i, j, n, m, k);
#ifdef DEBUG
            cout << "   (" << i << ", " << j << ") = " << index << "\n";
#endif
            cin >> M_host[index];
        }
    }

    // read matrix B

#ifdef DEBUG
    cout << "> read M_host part B\n";
#endif

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            int index = index_for_B(i, j, n, m, k);
#ifdef DEBUG
            cout << "   (" << i << ", " << j << ") = " << index << "\n";
#endif
            cin >> M_host[index];
        }
    }

#ifdef DEBUG
    // print matrices
    cout << "-------------------------- MATRIX M --------------------------\n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < (m + k); j++) {
            if (j == m) {
                cout << "| ";
            }
            int index = index_for_A(i, j, n, m, k);
            cout << M_host[index] << " ";
        }
        cout << "\n";
    }
    cout << "---------------------------------------------------------------\n\n";
#endif

    // X = [{X_row_1},...,{X_row_m}]

    int X_size = m * k;

#ifdef DEBUG
    cout << "> alloc memory for X_host\n";
#endif

    double *X_host = (double *)malloc(X_size * sizeof(double));

    // init matrix X
#ifdef DEBUG
    cout << "> init X_host\n";
#endif

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            int index = index_for_X(i, j, n, m, k);
#ifdef DEBUG
            cout << "   (" << i << ", " << j << ") = " << index << "\n";
#endif
            X_host[index] = 0.0;
        }
    }

    // Begin solving equation

    double *M_device; // matrix M on device
    double *X_device; // matrix X on device

    // alloc memory on device for matrix M

#ifdef DEBUG
    cout << "> CUDA alloc memory for M_device\n";
    cout << ">    M_size = " << M_size << "\n";
#endif

    CSC(cudaMalloc((void **)&M_device, M_size * sizeof(double)));
    cudaThreadSynchronize();

    // alloc memory on device for matrix X

#ifdef DEBUG
    cout << "> CUDA alloc memory for X_device\n";
#endif

    CSC(cudaMalloc((void **)&X_device, X_size * sizeof(double)));
    cudaThreadSynchronize();

    // copy matrix data

#ifdef DEBUG
    cout << "> CUDA copy M_host to M_device\n";
#endif

    CSC(cudaMemcpy(M_device, M_host, M_size * sizeof(double), cudaMemcpyHostToDevice));
    cudaThreadSynchronize();

    CSC(cudaMemcpy(X_device, X_host, X_size * sizeof(double), cudaMemcpyHostToDevice));
    cudaThreadSynchronize();

    int *prm_device; // rows permutations array on device

    // alloc memory for permutations array

#ifdef DEBUG
    cout << "> CUDA alloc memory for prm_device\n";
#endif

    CSC(cudaMalloc((void **)&prm_device, n * sizeof(int)));
    cudaThreadSynchronize();

    // init permutations array on device

#ifdef DEBUG
    cout << "> CUDA call init sequence for prm_device\n";
#endif

    initSequence <<<32, 32>>> (prm_device, n);
    cudaThreadSynchronize();

#ifdef DEBUG
    // check permutations init
    int *prm_host = (int *)malloc(n * sizeof(int));
    CSC(cudaMemcpy(prm_host, prm_device, n * sizeof(int), cudaMemcpyDeviceToHost));
    cudaThreadSynchronize();
    cout << "\n--------------------- PERMUTATIONS ARRAY --------------------\n";
    for (int i = 0; i < n; i++) {
        cout << prm_host[i] << " ";
    }
    cout << "\n-------------------------------------------------------------\n\n";
#endif

    int *x_index_device; // array with indexes for diagonal elements on device

    // alloc memory for array with indexes
    CSC(cudaMalloc((void **)&x_index_device, n * sizeof(int)));
    cudaThreadSynchronize();

    // init indexes on device
    initSequence <<<32, 32>>> (x_index_device, n);
    cudaThreadSynchronize();

#ifdef DEBUG
    int *x_index_host = (int *)malloc(n * sizeof(int));
    CSC(cudaMemcpy(x_index_host, x_index_device, n * sizeof(int), cudaMemcpyDeviceToHost));
    cudaThreadSynchronize();
    cout << "---------------------- x_index after init----------------------\n";
    for (int i = 0; i < n; i++) {
        cout << x_index_host[i] << " ";
    }
    cout << "\n-------------------------------------------------------------\n";
#endif

    int row = 0; // current row
    int col = 0; // current column

    int *max_value_row_device; // pointer to max value row on device
    CSC(cudaMalloc((void **)&max_value_row_device, sizeof(int)));
    cudaThreadSynchronize();

    for ( ; row < n && col < m; row++, col++) {

        // find row with max value
        findMaxValueRow <<<1, 1>>> (M_device, prm_device, row, col, n, m, k, max_value_row_device);
        cudaThreadSynchronize();

#ifdef DEBUG
        cout << "> Matrix M before swap rows\n";
        printMatrix <<<1, 1>>> (M_device, prm_device, n, m, k);
        cudaThreadSynchronize();
#endif

        // swap rows on device
        swap_rows <<<1, 1>>> (prm_device, row, max_value_row_device);
        cudaThreadSynchronize();

#ifdef DEBUG
        cout << "> Matrix M after swap rows\n";
        printMatrix <<<1, 1>>> (M_device, prm_device, n, m, k);
        cudaThreadSynchronize();
#endif

        // copy pivot value from device to host
        // int max_value_row_host;
        // CSC(cudaMemcpy(&max_value_row_host, max_value_row_device, sizeof(int), cudaMemcpyDeviceToHost));
        // cudaThreadSynchronize();

        int pivot_index_prm;
        CSC(cudaMemcpy(&pivot_index_prm, &prm_device[row], sizeof(int), cudaMemcpyDeviceToHost));
        cudaThreadSynchronize();

        double M_pivot_host;
        int M_pivot_host_index = index_for_A(pivot_index_prm, col, n, m, k);
        CSC(cudaMemcpy(&M_pivot_host, &M_device[M_pivot_host_index], sizeof(double), cudaMemcpyDeviceToHost));
        cudaThreadSynchronize();

#ifdef DEBUG
        printf("\n>> Position: row = %d, col = %d\n", row, col);
        printf(">> Pivot value = %f\n\n", M_pivot_host);
#endif

        if (fabs(M_pivot_host) > eps) { // non-zero pivot value
            updateRowsBelow <<<32, 32>>> (M_device, prm_device, x_index_device, row, col, n, m, k);
            cudaThreadSynchronize();
        } else {
            row--; // need to perform next iter on current row
        }

#ifdef DEBUG
        cout << "> Matrix M after updating rows\n";
        printMatrix <<<1, 1>>> (M_device, prm_device, n, m, k);
        cudaThreadSynchronize();
#endif
    }

    // last row position fix
    if (row == n || col == m) {
        row--;
    }

    // perform Gauss back substitution
    backSubstitution <<<32, 32>>> (M_device, X_device, prm_device, x_index_device, row, n, m, k);
    cudaThreadSynchronize();

    // copy matrix X from device to host
    CSC(cudaMemcpy(X_host, X_device, X_size * sizeof(double), cudaMemcpyDeviceToHost));
    cudaThreadSynchronize();

    // print X matrix
    print_matrix(X_host, n, m, k);

    // free device memory
    CSC(cudaFree(M_device));
    CSC(cudaFree(X_device));
    CSC(cudaFree(prm_device));
    CSC(cudaFree(x_index_device));

    // free host memory
    free(M_host);
    free(X_host);

    // done.
    return 0;
}
