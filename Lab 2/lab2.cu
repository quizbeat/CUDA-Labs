//
//  main.cpp
//  CUDA-Lab-2
//
//  Created by Nikita Makarov on 05/05/16.
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
#define IDX(i, j, w, h) (MAX(MIN(i, h - 1), 0) * (4 * w) + (4 * MAX(MIN(j, w - 1), 0)))
#define GRAYSCALE(image, pos) (0.299 * (*(image + pos))) + (0.587 * (*(image + pos + 1))) + (0.114 * (*(image + pos + 2)))

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

__global__ void SobelOperator(unsigned char *image, unsigned char *filtered, int w, int h) {

    int thread_i = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_j = blockIdx.y * blockDim.y + threadIdx.y;
    int offset_i = gridDim.x * blockDim.x;
    int offset_j = gridDim.y * blockDim.y;

    for (int i = thread_i ; i < h; i += offset_i) {
        for (int j = thread_j; j < w; j += offset_j) {

            double G_x_component_1 = 0.0;
            double G_x_compinent_2 = 0.0;

            for (int p = -1; p < 2; p++) {

                int pixel_index_top = IDX(i - 1, j + p, w, h);
                int pixel_index_bottom = IDX(i + 1, j + p, w, h);

                double gray_top = GRAYSCALE(image, pixel_index_top);
                double gray_bottom = GRAYSCALE(image, pixel_index_bottom);

                if (p == 0) {
                    gray_top *= 2;
                    gray_bottom *= 2;
                }

                G_x_component_1 += gray_bottom;
                G_x_compinent_2 += gray_top;
            }

            double G_x = G_x_component_1 - G_x_compinent_2;

            double G_y_component_1 = 0;
            double G_y_compinent_2 = 0;

            for (int p = -1; p < 2; p++) {

                int pixel_index_left = IDX(i + p, j - 1, w, h);
                int pixel_index_right = IDX(i + p, j + 1, w, h);

                double gray_left = GRAYSCALE(image, pixel_index_left);
                double gray_right = GRAYSCALE(image, pixel_index_right);

                if (p == 0) {
                    gray_left *= 2;
                    gray_right *= 2;
                }

                G_y_component_1 += gray_right;
                G_y_compinent_2 += gray_left;
            }

            double G_y = G_y_component_1 - G_y_compinent_2;

            double sobel = sqrt((G_x * G_x) + (G_y * G_y));
            unsigned char normalized_sobel = MAX(MIN(sobel, 255), 0);

            filtered[IDX(i, j, w, h) + 0] = normalized_sobel;
            filtered[IDX(i, j, w, h) + 1] = normalized_sobel;
            filtered[IDX(i, j, w, h) + 2] = normalized_sobel;
            filtered[IDX(i, j, w, h) + 3] = 0;
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

    unsigned char *filtered = (unsigned char *)malloc(4 * w * h * sizeof(unsigned char));
    unsigned char *filtered_device = NULL;
    CSC(cudaMalloc((void **)&filtered_device, 4 * w * h * sizeof(unsigned char)));
    CSC(cudaMemcpy(filtered_device, filtered, 4 * w * h * sizeof(unsigned char), cudaMemcpyHostToDevice));

    dim3 gridSize(32, 32);
    dim3 blockSize(32, 32);

    SobelOperator <<<gridSize, blockSize>>> (image_device, filtered_device, w, h);
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(filtered, filtered_device, 4 * w * h * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    write_image_to_file(filtered, w, h, output_path.c_str());

    CSC(cudaFree(image_device));
    CSC(cudaFree(filtered_device));

    free(image);
    free(filtered);

    return 0;
}
