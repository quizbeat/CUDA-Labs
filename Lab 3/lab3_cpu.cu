//
//  lab3_cpu.cu
//  CUDA-Lab-3
//
//  Created by Nikita Makarov on 07/05/16.
//  Copyright © 2016 Nikita Makarov. All rights reserved.
//

#include <iostream>
#include <limits>
#include <cfloat>
#include <stdio.h>
#include <math.h>


#define MIN(a, b) (a < b ? a : b)
#define MAX(a, b) (a > b ? a : b)
#define EPS 1e-7
#define BUCKET_SIZE 5
#define SPLIT_SIZE  2



// =============================================================================
//                          PRINT ARRAY FUNCTIONS
// =============================================================================
void print_array(float *data, int size) {
    for (int i = 0; i < size; i++) {
        printf("%f ", data[i]);
    }
    printf("\n");
}

void print_subarray(float *data, int begin, int end) {
    for (int i = begin; i < end; i++) {
        printf("%f ", data[i]);
    }
    printf("\n");
}



// =============================================================================
//                       DATA READ/WRITE FUNCTIONS
// =============================================================================
float *read_data(int *n) {
    fread(n, sizeof(int), 1, stdin);
    float *data = (float *)malloc(*n * sizeof(float));
    fread(data, sizeof(float), *n, stdin);
    return data;
}

void write_data(float *data, int n) {
    fwrite(data, sizeof(float), n, stdout);
}

void write_data_with_size(float *data, int n) {
    fwrite(&n, sizeof(int), 1, stdout);
    fwrite(data, sizeof(float), n, stdout);
}



// =============================================================================
//                              MAIN ALGORITHMS
// =============================================================================

// map float value to split index
int indexFromFloatValue(float value, float min, float max, int splits_count) {
    int index = (int)((value - min) / (max - min) * (splits_count - 1));
    return index;
}

void reduce_min_max(float *data, int size, float *min, float *max) {
    *min =  FLT_MAX;
    *max = -FLT_MAX;
    for (int i = 0; i < size; i++) {
        // ugly coding style?..
        if (data[i] > *max) { *max = data[i]; }
        if (data[i] < *min) { *min = data[i]; }
    }
}

void scan() {

}

void histogram() {

}



// =============================================================================
//                                   SORTING
// =============================================================================
void odd_even_sort(float *data, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = i & 1; j < size - 1; j += 2) {
            if (data[j] > data[j + 1]) {
                std::swap(data[j], data[j + 1]);
            }
        }
    }
}

void odd_even_sort_range(float *data, int begin, int end) {
    int size = end - begin;
    for (int i = 0; i < size; i++) {
        for (int j = i & 1; j < size - 1; j += 2) {
            if (data[j + begin] > data[j + begin + 1]) {
                std::swap(data[j + begin], data[j + begin + 1]);
            }
        }
    }
}

void bucket_sort(float *data, int size) {

    printf(">>> START BUCKET SORT\n");

    // STEP 1: Find min and max value
    float min = FLT_MAX;
    float max = -FLT_MAX;
    for (int i = 0; i < size; i++) {
        if (data[i] > max) {
            max = data[i];
        }
        if (data[i] < min) {
            min = data[i];
        }
    }

    // max += 1.f;
    // if (min > EPS) {
    //     min = 0.f;
    // }

#ifdef DEBUG
    printf("MIN = %f\n", min);
    printf("MAX = %f\n", max);
#endif

    // // normalize data
    // for (int i = 0; i < size; i++) {
    //     data[i] -= min; // shift to right
    //     data[i] /= max; // map to [0..1] range
    // }

// #ifdef DEBUG
//     printf("NORMALIZED DATA: ");
//     print_array(data, size);
// #endif


    // STEP 2: Make splits

    // calculate splits count
    int splits_count = size / 2 + 1;

    // create array for splits sizes
    int *size_of_split = (int *)malloc(splits_count * sizeof(int));
    for (int i = 0; i < splits_count; i++) {
        size_of_split[i] = 0;
    }

    // create splits array
    float **splits = (float **)malloc(splits_count * sizeof(float *));
    for (int i = 0; i < splits_count; i++) {
        splits[i] = (float *)malloc(size * sizeof(float));
    }

    // fill splits
    for (int i = 0; i < size; i++) {
        int index = indexFromFloatValue(data[i], min, max, splits_count);
        int index_in_split = size_of_split[index];
        splits[index][index_in_split] = data[i];
        size_of_split[index]++;
#ifdef DEBUG
        printf("value [%f] to splint #[%d]\n", data[i], index);
#endif
    }

    abort();




    // make buckets
    int buckets_count = 3;
    int *buckets_size = (int *)malloc(buckets_count * sizeof(int));
    for (int i = 0; i < buckets_count; i++) {
        buckets_size[i] = 0;
    }
    float **buckets = (float **)malloc(buckets_count * sizeof(float *));
    for (int i = 0; i < buckets_count; i++) {
        buckets[i] = (float *)malloc(size * sizeof(float));
    }



    // fill buckets
    for (int i = 0; i < size; i++) {
#ifdef DEBUG
        printf("Item #%d (%f)\n", i, data[i]);
#endif
        int bucket_index = (int)floor(data[i] * buckets_count);
#ifdef DEBUG
        printf(" bucket_index = %d\n", bucket_index);
#endif
        int item_position = buckets_size[bucket_index];
        buckets[bucket_index][item_position] = data[i];
        buckets_size[bucket_index]++;
    }

    // sort buckets
    for (int i = 0; i < buckets_count; i++) {
        int size = buckets_size[i];
        if (size > BUCKET_SIZE) {
            bucket_sort(buckets[i], size);
        }
        odd_even_sort(buckets[i], size);
    }

    // merge buckets
    int index = 0;
    for (int i = 0; i < buckets_count; i++) {
        int size = buckets_size[i];
        for (int j = 0; j < size; j++) {
#ifdef DEBUG
            printf("> merge (%d, %d)\n", i, j);
            printf("  index = %d\n", index);
            printf("   (before update) data[index] = %f\n", data[index]);
#endif
            float value = buckets[i][j] * max + min;
            data[index] = value;
#ifdef DEBUG
            printf("   (after update)  data[index] = %f\n", data[index]);
#endif
            index++;
        }
    }

    printf(">>> END BUCKET SORT\n");
}


// =============================================================================
//                                  MAIN
// =============================================================================
int main() {

    int n = 0;
    float *data = read_data(&n);
    bucket_sort(data, n);
    print_array(data, n);
    //write_data(data, n);

    return 0;
}
