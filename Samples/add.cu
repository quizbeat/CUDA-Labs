#include <stdio.h>


#define CSC(call) {														\
    cudaError err = call;												\
    if(err != cudaSuccess) {											\
        fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",	\
            __FILE__, __LINE__, cudaGetErrorString(err));				\
        exit(1);														\
    }																	\
} while (0)


__global__ void kernel(int *a, int *b, int *c, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = gridDim.x * blockDim.x;
	for(; idx < n; idx += offset)
		c[idx] = a[idx] + b[idx];
}

int main() {
	int i, n = 2000000;
	int *a = (int *)malloc(sizeof(int) * n);
	int *b = (int *)malloc(sizeof(int) * n);
	int *c = (int *)malloc(sizeof(int) * n);
	for(i = 0; i < n; i++)
		a[i] = b[i] = i;

	int *dev_a;
	int *dev_b;
	int *dev_c;
	cudaEvent_t start, stop;
	CSC(cudaEventCreate(&start));
	CSC(cudaEventCreate(&stop));

	CSC(cudaMalloc(&dev_a, sizeof(int) * n));
	CSC(cudaMalloc(&dev_b, sizeof(int) * n));
	CSC(cudaMalloc(&dev_c, sizeof(int) * n));

	CSC(cudaMemcpy(dev_a, a, sizeof(int) * n, cudaMemcpyHostToDevice));
	CSC(cudaMemcpy(dev_b, b, sizeof(int) * n, cudaMemcpyHostToDevice));
	CSC(cudaEventRecord(start, 0));
	//for(i = 0; i < n; i++)
	//	c[i] = a[i] + b[i];
	kernel<<<6, 256>>>(dev_a, dev_b, dev_c, n);
	CSC(cudaGetLastError());
	CSC(cudaEventRecord(stop, 0));
	CSC(cudaEventSynchronize(stop));
	float t;
	CSC(cudaEventElapsedTime(&t, start, stop));
	printf("time = %f\n", t);
	CSC(cudaEventDestroy(start));
	CSC(cudaEventDestroy(stop));

	CSC(cudaMemcpy(c, dev_c, sizeof(int) * n, cudaMemcpyDeviceToHost));

	//for(i = 0; i < n; i++)
	//	printf("%d ", c[i]);
	//printf("\n");

	CSC(cudaFree(dev_a));
	CSC(cudaFree(dev_b));
	CSC(cudaFree(dev_c));

	free(a);
	free(b);
	free(c);
	return 0;
}
