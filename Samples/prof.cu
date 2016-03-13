#include <stdio.h>

__global__ void kernel(double *a, int n, double k) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	int i, j;
	for(i = idx; i < n; i += offsetx)
		for(j = idy; j < n; j += offsety)
			a[j * n + i] *= k; 
}

int main() {
	int i, n = 1024;
	double *a = (double*)malloc(sizeof(double) * n * n);
	for(i = 0; i < n * n; i++)
		a[i] = i;
	double *dev_a;
	cudaMalloc(&dev_a, sizeof(double) * n * n);
	cudaMemcpy(dev_a, a, sizeof(double) * n * n, cudaMemcpyHostToDevice);
	kernel<<<dim3(16, 16), dim3(32, 8)>>>(dev_a, n, 2.3);
	 kernel<<<dim3(16, 16), dim3(32, 8)>>>(dev_a, n, 2.3);
	cudaMemcpy(a, dev_a, sizeof(double) * n * n, cudaMemcpyDeviceToHost);
// a...
//	for(i = 0; i < n * n; i++)
//		printf("%f ", a[i]);
//	printf("\n");
	cudaFree(dev_a);
	free(a);
	return 0;	
}

