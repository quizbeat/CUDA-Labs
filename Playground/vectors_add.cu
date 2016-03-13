#include <stdio.h>

#define CSC(call) {														                       \
    cudaError err = call;												                     \
    if(err != cudaSuccess) {											                   \
        fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n", \
            __FILE__, __LINE__, cudaGetErrorString(err));				     \
        exit(1);														                         \
    }																	                               \
} while (0)

float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;

bool noprompt = false;

void cleanup_resources();
void random_init(float *, int);
void parse_arguments(int, char **);

// kernel code
__global__ void vector_add(const float *A, const float *B, float *C, int N) {
  // get thread number
  int i = blockDim.x + blockIdx.x + threadIdx.x;
  if (i < N) {
    C[i] = A[i] + B[i];
  }
}

// host code
int main(int argc, char **argv) {
  int N = 50000;
  size_t size = N * sizeof(float);

  h_A = (float *)malloc(size);
  if (h_A == 0) {
    cleanup_resources();
  }

  h_B = (float *)malloc(size);
  if (h_B == 0) {
    cleanup_resources();
  }

  h_C = (float *)malloc(size);
  if (h_C == 0) {
    cleanup_resources();
  }

  random_init(h_A, N);
  random_init(h_B, N);

  // alloc memory on gpu
  CSC(cudaMalloc((void **)&d_A, size));
  CSC(cudaMalloc((void **)&d_B, size));
  CSC(cudaMalloc((void **)&d_C, size));

  // copy data to gpu
  CSC(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
  CSC(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

  // kernel call
  int threads_per_block = 256;
  int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

  vector_add<<<blocks_per_grid, threads_per_block>>> (d_A, d_B, d_C, N);

  // copy result to host
  CSC(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
  cleanup_resources();
}

void cleanup_resources() {
  if (d_A) {
    cudaFree(d_A);
  }
  if (d_B) {
    cudaFree(d_B);
  }
  if (d_C) {
    cudaFree(d_C);
  }
  if (h_A) {
    cudaFree(h_A);
  }
  if (h_B) {
    cudaFree(h_B);
  }
  if (h_C) {
    cudaFree(h_C);
  }
}

void random_init(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = rand() / (float)RAND_MAX;
  }
}
