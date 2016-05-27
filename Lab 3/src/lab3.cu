//
//  lab3.cu
//  CUDA-Lab-3
//
//  Created by Nikita Makarov on 07/05/16.
//  Copyright © 2016 Nikita Makarov. All rights reserved.
//

#include <iostream>
#include <stdio.h>
#include <limits>
#include <cfloat>
#include <math.h>
#include <time.h>
#include <fstream>

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

#define EPS 1e-7

// gpu defined properties
#define GRID_SIZE  32
#define BLOCK_SIZE 32

// some hacks
#define LOG_NUM_BANKS 5 // for 32
#define CONFLICT_FREE_OFFSET(i) ((i) >> LOG_NUM_BANKS)


// sort properties
#define BUCKET_SIZE 1024
#define SPLIT_SIZE  512


#define INDEX_FROM_FLOAT_VALUE(value,min,max,count) (int)((value-min)/(max-min)*(count-1))






// recursion depth control
int depth = 0;
int max_depth = 0;

void depth_inc() {
    depth++;
    if (depth > max_depth) {
        max_depth = depth;
    }
}

void depth_dec() {
    depth--;
}

void print_depth_bar() {
    for (int i = 0; i < depth; i++) {
        printf("__");
    }
}




// =============================================================================
//                                   PRINT
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

void print_array(int *data, int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", data[i]);
    }
    printf("\n");
}

void print_int_address(int *p) {
    printf(">>>> INT   Address = %p,  Value = %d\n", p, *p);
}

void print_float_address(float *p) {
    printf(">>>> FLOAT Address = %p,  Value = %f\n", p, *p);
}




// =============================================================================
//                              DATA READ/WRITE
// =============================================================================

float *read_data(int *n) {
    fread(n, sizeof(int), 1, stdin);
    float *data = (float *)malloc(*n * sizeof(float));
    fread(data, sizeof(float), *n, stdin);
    return data;
}

float *read_data_as_plain_text(int *n) {
    scanf("%d", n);
    float *data = (float *)malloc(*n * sizeof(float));
    int count = 0;
    for (int i = 0; i < *n; i++) {
        scanf("%f", &data[i]);
        count++;
    }
    printf("data count: %d\n", count);
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
//                                    HELPERS
// =============================================================================

float max_float(float lhs, float rhs) {
    return (lhs > rhs) ? lhs : rhs;
}

float min_float(float lhs, float rhs) {
    return (lhs < rhs) ? lhs : rhs;
}

float sum_float(float lhs, float rhs) {
    return lhs + rhs;
}

float multiply_float(float lhs, float rhs) {
    return lhs * rhs;
}

int sum_int(int lhs, int rhs) {
    return lhs + rhs;
}

// map float value to split index
int index_from_float_value(float value, float min, float max, int splits_count) {
    int index = (int)((value - min) / (max - min) * (splits_count - 1));
    return index;
}

void swap(float *lhs, float *rhs) {
    float temp = *lhs;
    *lhs = *rhs;
    *rhs = temp;
}



// =============================================================================
//                                   REDUCE
// =============================================================================

// plain cpu reduce
void reduce(float *data, int n, float *result, float (*op)(float, float), float identity) {
    *result = identity;
    for (int i = 0; i < n; i++) {
        *result = op(data[i], *result);
    }
}

void reduce_min_max(float *data, int size, float *min, float *max) {
    *min =  FLT_MAX;
    *max = -FLT_MAX;
    for (int i = 0; i < size; i++) {
        if (data[i] > *max) { *max = data[i]; }
        if (data[i] < *min) { *min = data[i]; }
    }
}



// gpu reduce
__global__ void gpuReduceSumFloat(float *data, int n, float *result) {

    __shared__ float shared_data[2048];

    int global_thread_id = 2 * blockDim.x * blockIdx.x + threadIdx.x;
    int thread_id = threadIdx.x;

    if (global_thread_id + blockDim.x < n) {
        shared_data[thread_id] = data[global_thread_id] + data[global_thread_id + blockDim.x];
    } else if (global_thread_id < n) {
        shared_data[thread_id] = data[global_thread_id];
    } else {
        shared_data[thread_id] = 0;
    }

    __syncthreads();;

    // reduction process
    for (int i = blockDim.x / 2; i > 0; i /= 2) {
        if (thread_id < i) {
            shared_data[thread_id] += shared_data[thread_id + i];
        }
        __syncthreads();
    }

    // write result to global memory
    if (thread_id == 0) {
        result[blockIdx.x] = shared_data[0];
    }
}

__global__ void gpuReduceMaxFloat(float *data, int n, float *result) {

    __shared__ float shared_data[2048];

    int global_thread_id = 2 * blockDim.x * blockIdx.x + threadIdx.x;
    int thread_id = threadIdx.x;

    if (global_thread_id + blockDim.x < n) {
        shared_data[thread_id] = MAX(data[global_thread_id], data[global_thread_id + blockDim.x]);
    } else if (global_thread_id < n) {
        shared_data[thread_id] = data[global_thread_id];
    } else {
        shared_data[thread_id] = data[0]; // just dummy
    }

    __syncthreads();;

    // reduction process
    for (int i = blockDim.x / 2; i > 0; i /= 2) {
        if (thread_id < i) {
            shared_data[thread_id] = MAX(shared_data[thread_id], shared_data[thread_id + i]);
        }
        __syncthreads();
    }

    // write result to global memory
    if (thread_id == 0) {
        result[blockIdx.x] = shared_data[0];
    }
}

__global__ void gpuReduceMinFloat(float *data, int n, float *result) {

    __shared__ float shared_data[2048];

    int global_thread_id = 2 * blockDim.x * blockIdx.x + threadIdx.x;
    int thread_id = threadIdx.x;

    if (global_thread_id + blockDim.x < n) {
        shared_data[thread_id] = MIN(data[global_thread_id], data[global_thread_id + blockDim.x]);
    } else if (global_thread_id < n) {
        shared_data[thread_id] = data[global_thread_id];
    } else {
        shared_data[thread_id] = data[0]; // just dummy
    }

    __syncthreads();;

    // reduction process
    for (int i = blockDim.x / 2; i > 0; i /= 2) {
        if (thread_id < i) {
            shared_data[thread_id] = MIN(shared_data[thread_id], shared_data[thread_id + i]);
        }
        __syncthreads();
    }

    // write result to global memory
    if (thread_id == 0) {
        result[blockIdx.x] = shared_data[0];
    }
}

__host__ void recursive_gpu_reduce_sum(float *data_device, int n, float *result_host) {
    int gridSize = (n / 2048) + 1;
    int blockSize = 1024;

    // printf("gridSize = %d\n", gridSize);

    float *result_device = NULL;
    CSC(cudaMalloc((void **)&result_device, gridSize * sizeof(float)));
    CSC(cudaGetLastError());

    if (result_device == NULL) {
        printf("YOLO! NULL MEMORY DETECTED!\n");
    }

    gpuReduceSumFloat <<<gridSize, blockSize>>> (data_device, n, result_device);
    CSC(cudaGetLastError());

    if (gridSize > 1) {
        recursive_gpu_reduce_sum(result_device, gridSize, result_host);
    } else {
        CSC(cudaMemcpy(result_host, result_device, sizeof(float), cudaMemcpyDeviceToHost));
        CSC(cudaGetLastError());
    }
    CSC(cudaFree(result_device));
    CSC(cudaGetLastError());
}

__host__ void recursive_gpu_reduce_max(float *data_device, int n, float *result_host) {
    int gridSize = (n / 2048) + 1;
    int blockSize = 1024;

    // printf("gridSize = %d\n", gridSize);

    float *result_device = NULL;
    CSC(cudaMalloc((void **)&result_device, gridSize * sizeof(float)));
    CSC(cudaGetLastError());

    if (result_device == NULL) {
        printf("YOLO! NULL MEMORY DETECTED!\n");
    }

    gpuReduceMaxFloat <<<gridSize, blockSize>>> (data_device, n, result_device);
    CSC(cudaGetLastError());

    if (gridSize > 1) {
        recursive_gpu_reduce_max(result_device, gridSize, result_host);
    } else {
        CSC(cudaMemcpy(result_host, result_device, sizeof(float), cudaMemcpyDeviceToHost));
        CSC(cudaGetLastError());
    }
    CSC(cudaFree(result_device));
    CSC(cudaGetLastError());
}

__host__ void recursive_gpu_reduce_min(float *data_device, int n, float *result_host) {
    int gridSize = (n / 2048) + 1;
    int blockSize = 1024;

    // printf("gridSize = %d\n", gridSize);

    float *result_device = NULL;
    CSC(cudaMalloc((void **)&result_device, gridSize * sizeof(float)));
    CSC(cudaGetLastError());

    if (result_device == NULL) {
        printf("YOLO! NULL MEMORY DETECTED!\n");
    }

    printf("before reduce min: n = %d, gridSize = %d, blockSize = %d\n", n, gridSize, blockSize);
    gpuReduceMinFloat <<<gridSize, blockSize>>> (data_device, n, result_device);
    CSC(cudaGetLastError());

    if (gridSize > 1) {
        recursive_gpu_reduce_min(result_device, gridSize, result_host);
    } else {
        CSC(cudaMemcpy(result_host, result_device, sizeof(float), cudaMemcpyDeviceToHost));
        CSC(cudaGetLastError());
    }
    CSC(cudaFree(result_device));
    CSC(cudaGetLastError());
}






// =============================================================================
//                                    SCAN
// =============================================================================

// plain cpu scan
// floating point numbers scan
void scan(float *data, int n, float *result_data, float (*op)(float, float), float identity, bool inclusive) {
    float result = identity;
    for (int i = 0; i < n; i++) {
        if (inclusive) {
            result = op(result, data[i]);
            result_data[i] = result;
        } else { // exclusive
            result_data[i] = result;
            result = op(result, data[i]);
        }
    }
}

// integer numbers scan
void scan(int *data, int n, int *result_data, int (*op)(int, int), int identity, bool inclusive) {
    int result = identity;
    for (int i = 0; i < n; i++) {
        if (inclusive) {
            result = op(result, data[i]);
            result_data[i] = result;
        } else { // exclusive
            result_data[i] = result;
            result = op(result, data[i]);
        }
    }
}

void scan_calculate_positions(int *position_for_split, int n, int *size_of_split) {
    for (int i = 1; i < n; i++) {
        position_for_split[i] = position_for_split[i - 1] + size_of_split[i - 1];
    }
}


// gpu scan
// Blelloch Scan
__global__ void BlellochScanSumFloat(float *data, int n, float *result) {
    __shared__ float shared_data[2 * BLOCK_SIZE];

    int thread_id = threadIdx.x;
    int offset = 1;

    shared_data[thread_id + BLOCK_SIZE] = data[thread_id + BLOCK_SIZE];

    for (int i = n / 2; i > 0; i /= 2) {
        __syncthreads();
        if (thread_id < i) {
            int ai = offset * (2 * thread_id + 1) - 1;
            int bi = offset * (2 * thread_id + 2) - 1;
            shared_data[bi] += shared_data[ai];
        }
        offset *= 2;
    }

    // clear last item
    if (thread_id == 0) {
        shared_data[n - 1] = 0; // ????
    }

    // copying and addition
    for (int i = 1; i < n; i *= 2) {
        offset /= 2;
        __syncthreads();
        if (thread_id < i) {
            int ai = offset * (2 * thread_id + 1) - 1;
            int bi = offset * (2 * thread_id + 2) - 1;
            float t = shared_data[ai];
            shared_data[ai] = shared_data[bi];
            shared_data[bi] += t;
        }
    }

    __syncthreads();

    // write result
    result[2 * thread_id] = shared_data[2 * thread_id];
    result[2 * thread_id + 1] = shared_data[2 * thread_id + 1];
}

__global__ void scan2(float *data, int n, float *result) {
    __shared__ float temp[2 * BLOCK_SIZE + CONFLICT_FREE_OFFSET(2 * BLOCK_SIZE)];

    int thread_id = threadIdx.x;
    int offset = 1;
    int ai = thread_id;
    int bi = thread_id + (n / 2);
    int offset_A = CONFLICT_FREE_OFFSET(ai);
    int offset_B = CONFLICT_FREE_OFFSET(bi);

    temp[ai + offset_A] = data[ai];
    temp[bi + offset_B] = data[bi];

    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (thread_id < d) {
            int ai = offset * (2 * thread_id + 1) - 1;
            int bi = offset * (2 * thread_id + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            temp[bi] += temp[ai];
        }
        offset <<= 1;
    }

    if (thread_id == 0) {
        temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
    }

    for (int d = 1; d < n; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (thread_id < d) {
            int ai = offset * (2 * thread_id + 1) - 1;
            int bi = offset * (2 * thread_id + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }

    __syncthreads();

    result[ai] = temp[ai + offset_A];
    result[bi] = temp[bi + offset_B];
}

__global__ void scan3(int *data, int n, int *sums, int *result) {
    __shared__ int temp[2 * BLOCK_SIZE + CONFLICT_FREE_OFFSET(2 * BLOCK_SIZE)];

    int thread_id = threadIdx.x;
    int offset = 1;
    int ai = thread_id;
    int bi = thread_id + (n / 2);  // different with Roma's code
    int offset_A = CONFLICT_FREE_OFFSET(ai);
    int offset_B = CONFLICT_FREE_OFFSET(bi);

    // printf("scan3 [ai + offset_A] = [%d]\nscan3 [bi + offset_B] = [%d]\n", ai + offset_A, bi + offset_B);

    temp[ai + offset_A] = data[ai + 2 * BLOCK_SIZE * blockIdx.x];
    temp[bi + offset_B] = data[bi + 2 * BLOCK_SIZE * blockIdx.x];

    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (thread_id < d) {
            int ai = offset * (2 * thread_id + 1) - 1;
            int bi = offset * (2 * thread_id + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            temp[bi] += temp[ai];
        }
        offset <<= 1;
    }

    if (thread_id == 0) {
        int idx = n - 1 + CONFLICT_FREE_OFFSET(n - 1);
        sums[blockIdx.x] = temp[idx];
        temp[idx] = 0;
    }

    for (int d = 1; d < n; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (thread_id < d) {
            int ai = offset * (2 * thread_id + 1) - 1;
            int bi = offset * (2 * thread_id + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }

    __syncthreads();

    offset = 2 * BLOCK_SIZE * blockIdx.x;

    // printf("scan3 indexes = (%d, %d)\n", ai + offset, bi + offset);

    result[ai + offset] = temp[ai + offset_A];
    result[bi + offset] = temp[bi + offset_B];
}

__global__ void scanDistribute(int *data, int *sums) {
    int idx = threadIdx.x + blockIdx.x * 2 * BLOCK_SIZE;
    // printf("scanDistribute index = %d\n", idx);
    data[idx] += sums[blockIdx.x];
}

__host__ void recursive_gpu_scan(int *data, int n, int *result) {

    printf("--- recursive_gpu_scan (data size = %d)\n", n);

    int numBlocks = n / (2 * BLOCK_SIZE) + 1;
    // if (numBlocks < 1) {
    //     numBlocks = 1;
    // }
    printf("numBlocks = %d\n", numBlocks);

    int *sums  = NULL;
    int *sums2 = NULL;

    CSC(cudaMalloc((void **)&sums, numBlocks * sizeof(int)));
    CSC(cudaGetLastError());

    CSC(cudaMalloc((void **)&sums2, numBlocks * sizeof(int)));
    CSC(cudaGetLastError());


    // CSC(cudaMemset(sums, 0.f, numBlocks * sizeof(float)));
    // CSC(cudaMemset(sums2, 0.f, numBlocks * sizeof(float)));

    dim3 threads(BLOCK_SIZE, 1, 1);
    dim3 blocks(numBlocks, 1, 1);

    scan3 <<<blocks, threads>>> (data, 2 * BLOCK_SIZE, sums, result);
    CSC(cudaGetLastError());


    // float *result_host = (float *)malloc(n * sizeof(float));
    // CSC(cudaMemcpy(result_host, result, n * sizeof(float), cudaMemcpyDeviceToHost));
    // printf("result after scan3\n");
    // print_array(result_host, n);


    if (n >= 2 * BLOCK_SIZE) {
        // printf("%d >= 2 * %d\n", n, BLOCK_SIZE);
        recursive_gpu_scan(sums, numBlocks, sums2);
        CSC(cudaGetLastError());
    } else {
        CSC(cudaMemcpy(sums2, sums, numBlocks * sizeof(int), cudaMemcpyDeviceToDevice));
        CSC(cudaGetLastError());
    }

    if (numBlocks > 1) {
        threads = dim3(2 * BLOCK_SIZE, 1, 1);
        blocks = dim3(numBlocks - 1, 1, 1);

        printf("before distribute: blocks = %d, threads = %d\n", blocks.x, threads.x);
        scanDistribute <<<blocks, threads>>> (result + (2 * BLOCK_SIZE), sums2 + 1);
        CSC(cudaGetLastError());
    }

    // CSC(cudaMemcpy(result_host, result, n * sizeof(float), cudaMemcpyDeviceToHost));
    // printf("result after scanDistribute\n");
    // print_array(result_host, n);

    cudaFree(sums);
    CSC(cudaGetLastError());

    cudaFree(sums2);
    CSC(cudaGetLastError());
}


// =============================================================================
//                                  HISTOGRAM
// =============================================================================

void histogram(float *data, int n, int *result, int op(float value, float min, float max, int count), float min, float max, int count) {
    for (int i = 0; i < n; i++) {
        int index = op(data[i], min, max, count);
        result[index]++;
    }
}

void histogram_splits_sizes(float *data, int n, int *size_of_split, float min, float max, int splits_count) {
    for (int i = 0; i < n; i++) {
        int index = index_from_float_value(data[i], min, max, splits_count);
        size_of_split[index]++;
    }
}

void histogram_fill_splits(float *data, int n, float *splits, int *begin_postion_for_split, int *current_size_of_split, float min, float max, int splits_count) {
    for (int i = 0; i < n; i++) {
        float value = data[i];
        int split_index = index_from_float_value(value, min, max, splits_count);
        int insert_position = begin_postion_for_split[split_index] + current_size_of_split[split_index];
        splits[insert_position] = value;
        current_size_of_split[split_index]++;
    }
}


// gpu histogram
__global__ void gpuHistogramCalculateSplitsSizes(float *data, int n, int *result, float min, float max, int splits_count) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;
    for ( ; index < n; index += offset) {
        int insertion_index = INDEX_FROM_FLOAT_VALUE(data[index], min, max, splits_count);
        atomicAdd(&(result[insertion_index]), 1);
    }
}

__global__ void gpuHistogramFillSplits(float *data_device, int n, float *splits_device,
                                       int *begin_position_for_split_device,
                                       unsigned int *current_size_of_split_device,
                                       float min, float max, int splits_count)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;
    for ( ; index < n; index += offset) {
        float value = data_device[index];
        int split_index = INDEX_FROM_FLOAT_VALUE(value, min, max, splits_count); // can I call this function from GPU?..
        int current_size_of_split = atomicInc(&(current_size_of_split_device[split_index]), 1); // WARNING! POTENTIAL ERROR!
        int insert_position = begin_position_for_split_device[split_index] + current_size_of_split;
        splits_device[insert_position] = value;
    }
}



// =============================================================================
//                                   SORT
// =============================================================================

void odd_even_sort(float *data, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = i & 1; j < size - 1; j += 2) {
            if (data[j] > data[j + 1]) {
                swap(&data[j], &data[j + 1]);
            }
        }
    }
}

void odd_even_sort_range(float *data, int begin, int end) {
    int size = end - begin;
    for (int i = 0; i < size; i++) {
        for (int j = i & 1; j < size - 1; j += 2) {
            if (data[j + begin] > data[j + begin + 1]) {
                swap(&data[j + begin], &data[j + begin + 1]);
            }
        }
    }
}

void bucket_sort(float *data, int n) {

    depth_inc();

    float min = FLT_MAX;
    float max = -FLT_MAX;

    reduce(data, n, &min, &min_float,  FLT_MAX);
    reduce(data, n, &max, &max_float, -FLT_MAX);

    if (fabs(min - max) < EPS) { // data already sorted
        depth_dec();
        return;
    }


    // STEP 2: Make splits

    // calculate splits count
    int splits_count = n / SPLIT_SIZE + 1;


    // create array for splits sizes
    int *size_of_split = (int *)malloc(splits_count * sizeof(int));
    for (int i = 0; i < splits_count; i++) {
        size_of_split[i] = 0;
    }


    // perform histogram to calculate splits sizes
    histogram(data, n, size_of_split, &index_from_float_value, min, max, splits_count);


    // calculate splits begin position
    int *begin_position_for_split = (int *)malloc(splits_count * sizeof(int));
    scan(size_of_split, splits_count, begin_position_for_split, &sum_int, 0, false);


    int *current_size_of_split = (int *)malloc(splits_count * sizeof(int));
    for (int i = 0; i < splits_count; i++) {
        current_size_of_split[i] = 0;
    }


    // Example:
    //                          (empty split)
    // size_of_split[i]:    3         v  2       5
    // splits looks like: [ a1 a2 a3 | | a4 a5 | a6 a7 a8 a9 a10 ]
    //                      ^            ^       ^
    //                 current_position_for_split[i] (initial state)


    // create splits array
    float *splits = (float *)malloc(n * sizeof(float)); // !!!!

    // fill splits with histogram
    histogram_fill_splits(data, n, splits, begin_position_for_split, current_size_of_split, min, max, splits_count);



    // STEP 3: Make buckets

    int buckets_count = splits_count;
    int *size_of_bucket = (int *)malloc(buckets_count * sizeof(int)); // how many buckets??

    for (int i = 0; i < buckets_count; i++) {
        size_of_bucket[i] = 0;
    }

    int *begin_position_for_bucket = (int *)malloc(buckets_count * sizeof(int));


    int bucket_index = 0;

    for (int split_index = 0; split_index < splits_count; split_index++) {

        int split_size = size_of_split[split_index];

        if (split_size > BUCKET_SIZE) {

            bucket_index++;

            // sort current split
            float *split = &splits[begin_position_for_split[split_index]];
            bucket_sort(split, split_size);

            // remember split as bucket
            begin_position_for_bucket[bucket_index] = begin_position_for_split[split_index];
            size_of_bucket[bucket_index] = size_of_split[split_index];
            bucket_index++;

        } else {
            int current_bucket_remaining_capacity = BUCKET_SIZE - size_of_bucket[bucket_index];
            if (split_size <= current_bucket_remaining_capacity) {
                // insert split to current bucket
                if (current_bucket_remaining_capacity == BUCKET_SIZE) {
                    begin_position_for_bucket[bucket_index] = begin_position_for_split[split_index];
                }
                size_of_bucket[bucket_index] += size_of_split[split_index];

            } else {
                // insert split to next bucket
                bucket_index++;
                begin_position_for_bucket[bucket_index] = begin_position_for_split[split_index];
                size_of_bucket[bucket_index] = size_of_split[split_index];
            }
        }
    }

    // determine correct buckets count
    if (size_of_bucket[bucket_index] == 0) {
        buckets_count = bucket_index;
    } else {
        buckets_count = bucket_index + 1;
    }



    // STEP 4: Sort buckets

    for (int i = 0; i < buckets_count; i++) {
        float *bucket = &splits[begin_position_for_bucket[i]];
        int bucket_size = size_of_bucket[i];
        odd_even_sort(bucket, bucket_size);
    }

    memcpy(data, splits, n * sizeof(float));

    depth_dec();
}




//
// gpu_bucket_sort description:
//      data -- initial array allocated for GPU usage;
//         n -- amount of items in data array;
//    result -- sorted data array allocated for CPU usage.
//
__host__ void gpu_bucket_sort(float *data_device, int n) {

    printf("BEGIN SORT\n");

    // find min data value
    float min = FLT_MAX;
    recursive_gpu_reduce_min(data_device, n, &min);
    CSC(cudaGetLastError());

    // find max data value
    float max = -FLT_MAX;
    recursive_gpu_reduce_max(data_device, n, &max);
    CSC(cudaGetLastError());

    // check for already sorted array
    if (fabs(min - max) < EPS) {
        return;
    }

    // make splits
    int splits_count = n / SPLIT_SIZE + 1;

    int *size_of_split = (int *)malloc(splits_count * sizeof(int));
    for (int i = 0; i < splits_count; i++) {
        size_of_split[i] = 0;
    }

    // create size_of_split on gpu
    int *size_of_split_device = NULL;
    CSC(cudaMalloc((void **)&size_of_split_device, splits_count * sizeof(int)));
    CSC(cudaGetLastError());

    // calculate splits sizes with histogram
    gpuHistogramCalculateSplitsSizes <<<GRID_SIZE, BLOCK_SIZE>>> (data_device, n, size_of_split_device, min, max, splits_count);
    CSC(cudaGetLastError());

    // calculate splits begin position with scan
    int *begin_position_for_split_device = NULL;
    CSC(cudaMalloc((void **)&begin_position_for_split_device, splits_count * sizeof(int)));
    CSC(cudaGetLastError());

    recursive_gpu_scan(size_of_split_device, n, begin_position_for_split_device);
    CSC(cudaGetLastError());

    // int *current_size_of_split = (int *)malloc(splits_count * sizeof(int));
    // for (int i = 0; i < splits_count; i++) {
    //     current_size_of_split[i] = 0;
    // }

    unsigned int *current_size_of_split_device = NULL;
    CSC(cudaMalloc((void **)&current_size_of_split_device, splits_count * sizeof(unsigned int)));
    CSC(cudaGetLastError());

    CSC(cudaMemset(current_size_of_split_device, 0, splits_count * sizeof(unsigned int)));
    CSC(cudaGetLastError());

    // create splits array
    float *splits_device = NULL;
    CSC(cudaMalloc((void **)&splits_device, n * sizeof(float)));
    CSC(cudaGetLastError());

    // fill splits with histogram
    gpuHistogramFillSplits <<<GRID_SIZE, BLOCK_SIZE>>> (data_device, n, splits_device,
                                                        begin_position_for_split_device,
                                                        current_size_of_split_device,
                                                        min, max, splits_count);
    CSC(cudaGetLastError());

    // make buckets
    int buckets_count = splits_count;
    int *size_of_bucket = (int *)malloc(buckets_count * sizeof(int));
    memset(size_of_bucket, 0, buckets_count * sizeof(int));

    int *begin_position_for_bucket = (int *)malloc(buckets_count * sizeof(int));

    int bucket_index = 0;

    for (int split_index = 0; split_index < splits_count; split_index++) {

        int split_size = 0;
        CSC(cudaMemcpy(&split_size, &(size_of_split_device[split_index]), sizeof(int), cudaMemcpyDeviceToHost));
        CSC(cudaGetLastError());

        int split_begin_position = 0;
        CSC(cudaMemcpy(&split_begin_position, &(begin_position_for_split_device[split_index]), sizeof(int), cudaMemcpyDeviceToHost));
        CSC(cudaGetLastError());

        printf("split_size = %d\n", split_size);
        printf("split_begin_position = %d\n", split_begin_position);

        if (split_size > BUCKET_SIZE) {

            printf(">>>>>>>>>>>>> YEP\n");

            bucket_index++;

            // sort current split
            float *split = &(splits_device[split_begin_position]); // gpu pointer
            gpu_bucket_sort(split, split_size);

            // remember split as bucket
            begin_position_for_bucket[bucket_index] = split_begin_position; // ????
            // size_of_bucket[bucket_index] = split_size; // ????
            size_of_bucket[bucket_index] = -1; // -1 indicates that bucket already sorted
            bucket_index++;

        } else {

            int current_bucket_remaining_capacity = BUCKET_SIZE - size_of_bucket[bucket_index];
            if (split_size <= current_bucket_remaining_capacity) {
                // insert split to current bucket
                if (current_bucket_remaining_capacity == BUCKET_SIZE) {
                    begin_position_for_bucket[bucket_index] = split_begin_position;
                }
                size_of_bucket[bucket_index] += split_size;

            } else {
                // insert split to next bucket
                bucket_index++;
                begin_position_for_bucket[bucket_index] = split_begin_position;
                size_of_bucket[bucket_index] = split_size;
            }
        }
    }

    // determine correct buckets count
    if (size_of_bucket[bucket_index] == 0) {
        buckets_count = bucket_index;
    } else {
        buckets_count = bucket_index + 1;
    }

    // sort buckets

    for (int i = 0; i < buckets_count; i++) {
        int bucket_size = size_of_bucket[i];
        if (bucket_size == -1) { // already sorted
            continue;
        }
        float *bucket = (float *)malloc(bucket_size * sizeof(float));
        int bucket_begin_position = begin_position_for_bucket[i];

        CSC(cudaMemcpy(bucket, &(splits_device[bucket_begin_position]), bucket_size * sizeof(float), cudaMemcpyDeviceToHost));
        CSC(cudaGetLastError());

        odd_even_sort(bucket, bucket_size);

        CSC(cudaMemcpy(&(splits_device[bucket_begin_position]), bucket, bucket_size * sizeof(float), cudaMemcpyHostToDevice));
        CSC(cudaGetLastError());
    }

    CSC(cudaMemcpy(data_device, splits_device, n * sizeof(float), cudaMemcpyDeviceToDevice));

    printf("END SORT\n");

}


__host__ void __bucket_sort(float *data, int n) {
    // prepare device data
    float *data_device = NULL;
    CSC(cudaMalloc((void **)&data_device, n * sizeof(float)));
    CSC(cudaMemcpy(data_device, data, n * sizeof(float), cudaMemcpyHostToDevice));
    CSC(cudaGetLastError());

    gpu_bucket_sort(data_device, n);

    CSC(cudaMemcpy(data, data_device, n * sizeof(float), cudaMemcpyDeviceToHost));
    CSC(cudaGetLastError());
}


// =============================================================================
//                                  MAIN
// =============================================================================

bool sorted(float *data, int n) {
    for (int i = 1; i < n; i++) {
        if (data[i] < data[i - 1]) {
            return false;
        }
    }
    return true;
}


int main() {

/*
test
10
3 -5 6 7 4 1 8 10 2 5
*/

    int n = 0;
    float *data = read_data_as_plain_text(&n);
    // float *data = read_data(&n);

    __bucket_sort(data, n);

    print_array(data, n);

    return 0;
}
