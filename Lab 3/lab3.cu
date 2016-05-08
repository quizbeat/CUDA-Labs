//
//  lab3.cu
//  CUDA-Lab-3
//
//  Created by Nikita Makarov on 07/05/16.
//  Copyright © 2016 Nikita Makarov. All rights reserved.
//

#include <iostream>
#include <stdio.h>
#include <math.h>

#define CSC(call) {														\
    cudaError err = call;												\
    if(err != cudaSuccess) {											\
        fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",	\
            __FILE__, __LINE__, cudaGetErrorString(err));				\
        exit(1);														\
    }																	\
} while (0)

#define MIN(a, b) (a < b ? a : b)
#define MAX(a, b) (a > b ? a : b)

// CUDA kernels

__global__ void reduce() {

}

__global__ void scan() {

}

__global__ void histogram() {

}

float *read_numbers(int *n) {
    fread(n, sizeof(int), 1, stdin);
    float *numbers = (float *)malloc(*n * sizeof(float));
    fread(numbers, sizeof(float), *n, stdin);
    return numbers;
}

int main() {

    dim3 reduceGridSize(32, 32);
    dim3 reduceBlockSize(32, 32);

    dim3 scanGridSize(32, 32);
    dim3 scanBlockSize(32, 32);

    dim3 histogramGridSize(32, 32);
    dim3 histogramBlockSize(32, 32);

    int n = 0;
    float *data = read_numbers(&n);



    return 0;
}
