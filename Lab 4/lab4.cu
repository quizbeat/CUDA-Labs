//
//  main.cpp
//  CUDA-Lab-4
//
//  Created by Nikita Makarov on 06/05/16.
//  Copyright © 2016 Nikita Makarov. All rights reserved.
//

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

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
#define IDX(i, j, w, h) (i * 4 * w + 4 * j)

__host__ unsigned char *read_image_at_path(const char *path, int *w, int *h) {
    FILE *image_file = fopen(path, "rb");

    if (!image_file) {
        printf("Error: can't open file.");
        return NULL;
    }

    // read width and height from file
    fread(w, sizeof(int), 1, image_file);
    fread(h, sizeof(int), 1, image_file);

    // alloc memory for pixels
    unsigned char *image = (unsigned char *)malloc(4 * (*w) * (*h) * sizeof(unsigned char));

    // read pixels
    fread(image, sizeof(unsigned char), 4 * (*w) * (*h), image_file);

    fclose(image_file);

    return image;
}

__host__ void write_image_to_file(unsigned char *image, int w, int h, const char *path) {
    FILE *image_file = fopen(path, "wb");

    // write dimensions
    fwrite(&w, sizeof(int), 1, image_file);
    fwrite(&h, sizeof(int), 1, image_file);

    // write pixels
    fwrite(image, sizeof(unsigned char), 4 * w * h, image_file);

    fclose(image_file);
}

__device__ int func(int r, int g, int b, int r_avg, int g_avg, int b_avg) {
    int r_comp = r - r_avg;
    int g_comp = g - g_avg;
    int b_comp = b - b_avg;
    return -1 * (r_comp * r_comp + g_comp * g_comp + b_comp * b_comp);
}

__constant__ unsigned char average_pixel_device[32 * 3];

__global__ void classifyPixels(unsigned char *image, int w, int h, int nc) {
    int thread_i = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_j = blockIdx.y * blockDim.y + threadIdx.y;
    int offset_i = gridDim.x * blockDim.x;
    int offset_j = gridDim.y * blockDim.y;

    for (int i = thread_i; i < h; i += offset_i) {
        for (int j = thread_j; j < w; j += offset_j) {
            unsigned char red   = image[IDX(i, j, w, h) + 0];
            unsigned char green = image[IDX(i, j, w, h) + 1];
            unsigned char blue  = image[IDX(i, j, w, h) + 2];
            int max = (-3 * 255 * 255) - 1;
            unsigned char class_index = 0;
            for (int c = 0; c < nc; c++) {
                int value = func(red, green, blue,
                                 average_pixel_device[c * 3 + 0],
                                 average_pixel_device[c * 3 + 1],
                                 average_pixel_device[c * 3 + 2]);
                if (value > max) {
                    max = value;
                    class_index = (unsigned char)c;
                }
            }
            image[IDX(i, j, w, h) + 3] = class_index;
        }
    }
}

int main(int argc, const char * argv[]) {

    std::string input_path;
    std::string output_path;

    std::cin >> input_path;
    std::cin >> output_path;

    int w, h;
    unsigned char *image = read_image_at_path(input_path.c_str(), &w, &h);

    unsigned char *image_device = NULL;
    CSC(cudaMalloc((void **)&image_device, 4 * w * h * sizeof(unsigned char)));
    CSC(cudaMemcpy(image_device, image, 4 * w * h * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CSC(cudaGetLastError());

    int nc;
    std::cin >> nc;

    unsigned char *average_pixel = (unsigned char *)malloc(3 * nc * sizeof(unsigned char));

    for (int i = 0; i < nc; i++) {
        long np;

        int red_sum = 0;
        int green_sum = 0;
        int blue_sum = 0;

        std::cin >> np;
        for (long j = 0; j < np; j++) {
            int pos_i, pos_j;
            std::cin >> pos_i >> pos_j;
            red_sum   += image[IDX(pos_j, pos_i, w, h) + 0];
            green_sum += image[IDX(pos_j, pos_i, w, h) + 1];
            blue_sum  += image[IDX(pos_j, pos_i, w, h) + 2];
        }

        int red_average = red_sum / np;
        int green_average = green_sum / np;
        int blue_average = blue_sum / np;

        average_pixel[i * 3 + 0] = red_average;
        average_pixel[i * 3 + 1] = green_average;
        average_pixel[i * 3 + 2] = blue_average;
    }

    CSC(cudaMemcpyToSymbol(average_pixel_device, average_pixel, 3 * nc * sizeof(unsigned char)));
    CSC(cudaGetLastError());
    CSC(cudaThreadSynchronize());

    dim3 gridSize(32, 32);
    dim3 blockSize(32, 32);

    classifyPixels <<<gridSize, blockSize>>> (image_device, w, h, nc);
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(image, image_device, 4 * w * h * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    CSC(cudaGetLastError());

    write_image_to_file(image, w, h, output_path.c_str());

    free(image);
    free(average_pixel);

    CSC(cudaFree(image_device));

    return 0;
}
