#include <stdio.h>

#define CSC(call) {														\
    cudaError err = call;												\
    if(err != cudaSuccess) {											\
        fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",	\
            __FILE__, __LINE__, cudaGetErrorString(err));				\
        exit(1);														\
    }																	\
} while (0)

__global__ void kernel(int *a, int n, int k) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;
	for(; idx < n; idx += offset)
		a[idx] *= k;
}

int main() {
	int i, n = 10000;
	int *a = (int *)malloc(sizeof(int) * n);
	int *dev_a;
	for(i = 0; i < n; i++)
		a[i] = 1;
	CSC(cudaMalloc(&dev_a, sizeof(int) * n));
	CSC(cudaMemcpy(dev_a, a, sizeof(int) * n, cudaMemcpyHostToDevice));
	kernel<<<dim3(2), dim3(32)>>>(dev_a, n, 2);
	CSC(cudaGetLastError());
	CSC(cudaMemcpy(a, dev_a, sizeof(int) * n, cudaMemcpyDeviceToHost));
	for(i = 0; i < n; i++)
		printf("%d ", a[i]);
	printf("\n");
	CSC(cudaFree(dev_a));
	free(a);
	return 0;
}
