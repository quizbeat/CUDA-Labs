//
//  lab_1_7.cu
//  Matrix Equation Solver
//
//  Created by Nikita Makarov on 22/03/16.
//  Copyright © 2016 Nikita Makarov. All rights reserved.
//

#include "stdio.h"

#define CSC(call) {                                                     \
	 cudaError err = call;                                              \
	 if(err!=cudaSuccess) {                                             \
		 fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",   \
            __FILE__, __LINE__, cudaGetErrorString(err));				\
	 }                                                                  \
 } while (0)

 static const float eps = 1e-8;

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

__global__ void init_permutations(long *prm, long n)
{
    long index = blockDim.x * blockIdx.x + threadIdx.x;
	prm[index] = index;
}

__global__ void init_with_zeros(float **A) {

}

__host__ int main()
{
	long n, m, k;
	scanf("%d %d %d\n", &n, &m, &k);

	float **host_A = (float **)malloc(n * sizeof(float *));
	for (long i = 0; i < n; i++) {
        host_A[i] = (float *)malloc(m * sizeof(float));
    }

	for (long i = 0; i < n; i++) {
		for (long j = 0; j < m; j++) {
			scanf("%f", &host_A[i][j]);
		}
	}

	float **host_B = (float **)malloc(n *sizeof(float *));
	for (long i = 0; i < n; i++) {
        host_B[i] = (float *)malloc(k * sizeof(float));
    }

	for (long i = 0; i < n; i++) {
        for (long j = 0; j < k; j++) {
            scanf("%f", &host_B[i][j]);
        }
    }

	float **host_X = (float **)malloc(m * sizeof(float *));
    for (long i = 0; i < m; i++) {
        host_X[i] = (float *)malloc(k * sizeof(float));
    }

    // init matrix X (on device)
    for (long i = 0; i < m; i++) {
        for (long j = 0; j < k; j++) {
            X[i][j] = 0.0;
        }
    }



	return 0;
}
