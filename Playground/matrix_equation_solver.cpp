#include <iostream>
#include <cstdlib>

using namespace std;

void print_matrix(float **M, long n, long m)
{
    for (long i = 0; i < n; i++) {
        for (long j = 0; j < m; j++) {
            cout << M[i][j] << " ";
        }
        cout << endl;
    }
}

void swap_rows(long *permutations, long i, long j)
{
    long temp = permutations[i];
    permutations[i] = permutations[j];
    permutations[j] = temp;
}

void solve_equation(float **A, float **X, float **B, long m, long n, long k)
// solves matrix equation AX = B, where:
// A - n x m matrix, B - n x k matrix, X - unknown m x k matrix
{
    // create permutations array
    long *permutations = (long *)malloc(n * sizeof(long));
    for (long i = 0; i < n; i++) {
        permutations[i] = i;
    }

    // transform block matrix [A|B]
    for (long row = 0; row < n; row++) {
        long column = row;
        // find row with max value at current column
        long max_value_row = row;
        for (long i = row + 1; i < n; i++) {
            if (fabs(A[permutations[i]][column] > A[permutations[max_value_row]][column]) {
                max_value_row = i;
            }
        }
        // swap current row and found row
        swap_rows(permutations, row, max_value_row);

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

    return 0;
}
