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
#define SWAP_FLOATS(a,b) {float t = a; a = b; b = t;}





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

void print_depth_space() {
    for (int i = 0; i < depth; i++) {
        printf("  ");
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

void print_array(int *data, int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", data[i]);
    }
    printf("\n");
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
#ifdef DEBUG
    printf("data count: %d\n", count);
#endif

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
//                                   REDUCE
// =============================================================================

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
        print_depth_space();
        printf("YOLO! NULL MEMORY DETECTED!\n");
    }

#ifdef DEBUG
    print_depth_space();
    printf("before reduce min: n = %d, gridSize = %d, blockSize = %d\n", n, gridSize, blockSize);
#endif

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

__global__ void scan3(int *data, int n, int *sums, int *result) {
    // __shared__ int temp[2 * BLOCK_SIZE + CONFLICT_FREE_OFFSET(2 * BLOCK_SIZE)];

    __shared__ int temp[2 * 512 + CONFLICT_FREE_OFFSET(2 * 512)];

    int thread_id = threadIdx.x;
    int offset = 1;
    int ai = thread_id;
    // int bi = thread_id + (n / 2);  // different with Roma's code
    int bi = thread_id + 512;
    int offset_A = CONFLICT_FREE_OFFSET(ai);
    int offset_B = CONFLICT_FREE_OFFSET(bi);

    // printf("scan3 [ai + offset_A] = [%d]\nscan3 [bi + offset_B] = [%d]\n", ai + offset_A, bi + offset_B);

    temp[ai] = 0;
    temp[bi] = 0;

    __syncthreads();

    if (ai + 2 * 512 * blockIdx.x < n) {
        temp[ai + offset_A] = data[ai + 2 * 512 * blockIdx.x];
    } else {
        temp[ai + offset_A] = 0;
    }

    if (bi + 2 * 512 * blockIdx.x < n) {
        temp[ai + offset_B] = data[bi * 2 * 512 * blockIdx.x];
    } else {
        temp[bi + offset_B] = 0;
    }

    // temp[ai + offset_A] = data[ai + 2 * BLOCK_SIZE * blockIdx.x];
    // temp[bi + offset_B] = data[bi + 2 * BLOCK_SIZE * blockIdx.x];

    // for (int d = n >> 1; d > 0; d >>= 1) {
    for (int d = 512; d > 0; d >>= 1) {
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
        // int idx = n - 1 + CONFLICT_FREE_OFFSET(n - 1);
        int idx = 2 * 512 - 1 + CONFLICT_FREE_OFFSET(2 * 512 - 1);
        sums[blockIdx.x] = temp[idx];
        temp[idx] = 0;
    }

    __syncthreads();

    // for (int d = 1; d < n; d <<= 1) {
    for (int d = 1; d < 2 * 512; d <<= 1) {
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

    // offset = 2 * BLOCK_SIZE * blockIdx.x;
    offset = 2 * 512 * blockIdx.x;

    // printf("scan3 indexes = (%d, %d)\n", ai + offset, bi + offset);

    if (ai + offset < n) {
        result[ai + offset] = temp[ai + offset];
    }

    if (bi + offset < n) {
        result[bi + offset] = temp[bi + offset];
    }

    // result[ai + offset] = temp[ai + offset_A];
    // result[bi + offset] = temp[bi + offset_B];
}

__global__ void scanDistribute(int *data, int n, int *sums) {
    //int idx = threadIdx.x + blockIdx.x * 2 * BLOCK_SIZE;
    int idx = threadIdx.x + blockIdx.x * 2 * 512;
    if (idx < n) {
        data[idx] += sums[blockIdx.x];
    }
    // printf("scanDistribute index = %d\n", idx);
    // data[idx] += sums[blockIdx.x];
}

__host__ void recursive_gpu_scan(int *data, int n, int *result) {

#ifdef DEBUG
    print_depth_space();
    printf("recursive_gpu_scan (data size = %d)\n", n);
#endif

    // int numBlocks = n / (2 * BLOCK_SIZE) + 1;

    int threadsPerBlock = 512;
    int threads = 512 * 2;
    int numBlocks = n / (512 * 2) + 1;

#ifdef DEBUG
    print_depth_space();
    printf("numBlocks = %d\n", numBlocks);
#endif

    int *sums  = NULL;
    int *sums2 = NULL;

    CSC(cudaMalloc((void **)&sums, numBlocks * sizeof(int)));
    CSC(cudaMemset(sums, 0, numBlocks * sizeof(int)));
    CSC(cudaGetLastError());

    CSC(cudaMalloc((void **)&sums2, numBlocks * sizeof(int)));
    CSC(cudaMemset(sums2, 0, numBlocks * sizeof(int)));
    CSC(cudaGetLastError());

    // dim3 threads(BLOCK_SIZE, 1, 1);
    // dim3 blocks(numBlocks, 1, 1);

    // scan3 <<<blocks, threads>>> (data, 2 * BLOCK_SIZE, sums, result);
    scan3 <<<numBlocks, threadsPerBlock>>> (data, n, sums, result);
    CSC(cudaGetLastError());

    // if (n >= 2 * BLOCK_SIZE) {
    if (n >= threads) {
        // printf("%d >= 2 * %d\n", n, BLOCK_SIZE);
        recursive_gpu_scan(sums, numBlocks, sums2);
        CSC(cudaGetLastError());
    } else {
        CSC(cudaMemcpy(sums2, sums, numBlocks * sizeof(int), cudaMemcpyDeviceToDevice));
        CSC(cudaGetLastError());
    }

    // if (numBlocks > 1) {
    if (numBlocks - 1 > 0) {
        // threads = dim3(2 * BLOCK_SIZE, 1, 1);
        // blocks = dim3(numBlocks - 1, 1, 1);

        dim3 blocks(numBlocks - 1, 1, 1);
        dim3 threads(1024, 1, 1);

#ifdef DEBUG
        print_depth_space();
        printf("before distribute: blocks = %d, threads = %d\n", blocks.x, threads.x);
#endif

        // scanDistribute <<<blocks, threads>>> (result + (2 * BLOCK_SIZE), sums2 + 1);
        scanDistribute <<<blocks, threads>>> (result + 1024, n - 1024, sums2 + 1);

        CSC(cudaGetLastError());
    }

    cudaFree(sums);
    CSC(cudaGetLastError());

    cudaFree(sums2);
    CSC(cudaGetLastError());
}


// =============================================================================
//                                  HISTOGRAM
// =============================================================================

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
        int split_index = INDEX_FROM_FLOAT_VALUE(value, min, max, splits_count);
        // int current_size_of_split = atomicInc(&(current_size_of_split_device[split_index]), 1); // WARNING! POTENTIAL ERROR!
        int current_size_of_split = atomicAdd(&(current_size_of_split_device[split_index]), 1); /// ?????
        int insert_position = begin_position_for_split_device[split_index] + current_size_of_split;
        splits_device[insert_position] = value;
#ifdef DEBUG
        // printf("Block_id = %d, thread_id = %d -- insert value [%f] from index [%d] to splits index [%d]\n", blockIdx.x, threadIdx.x, value, index, insert_position);
#endif
    }
}



// =============================================================================
//                                   SORT
// =============================================================================

void swap(float *lhs, float *rhs) {
    float temp = *lhs;
    *lhs = *rhs;
    *rhs = temp;
}


void odd_even_sort(float *data, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = i & 1; j < size - 1; j += 2) {
            if (data[j] > data[j + 1]) {
                swap(&data[j], &data[j + 1]);
            }
        }
    }
}

// multiple threads
__global__ void oddEvenSort(float *data, int n, int buckets_count, int *begin_position_for_bucket, int *size_of_bucket) {
    int bucket_index = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;

    for (int b = bucket_index; b < buckets_count; b += offset) {
        int size = size_of_bucket[bucket_index];
        if (size == -1) { // already sorted
            continue;
        }
        int begin = begin_position_for_bucket[b];
        for (int i = begin; i < begin + size; i++) {
            for (int j = i & 1; j < begin + size - 1; j += 2) {
                if (data[j] > data[j + 1]) {
                    float temp = data[j];
                    data[j] = data[j + 1];
                    data[j + 1] = temp;
                }
            }
        }
    }
}

// 1 block - 1 bucket
__global__ void gpuOddEvenSort(float *buckets, int n, int *begin_position_for_bucket, int *size_of_bucket) {

    int bucket_index = blockIdx.x;
    int bucket_size = size_of_bucket[bucket_index];

    if (bucket_size == -1) { // bucket already sorted
        return;
    }

    // prepare shared array for bucket
    __shared__ float shared_bucket[BUCKET_SIZE];

    int thread_id = threadIdx.x;

    shared_bucket[2 * thread_id    ] = FLT_MAX; // dummy for item index out of bounds
    shared_bucket[2 * thread_id + 1] = FLT_MAX; // dummy for item index out of bounds

    __syncthreads();

    // load bucket items
    int item_index = 0;

    item_index = 2 * thread_id + begin_position_for_bucket[bucket_index];
    if (item_index - begin_position_for_bucket[bucket_index] < bucket_size) {
        shared_bucket[2 * thread_id] = buckets[item_index];
    }

    item_index = 2 * thread_id + 1 + begin_position_for_bucket[bucket_index];
    if (item_index - begin_position_for_bucket[bucket_index] < bucket_size) {
        shared_bucket[2 * thread_id + 1] = buckets[item_index];
    }

    __syncthreads();

    int  odd_index_limit = BUCKET_SIZE - 1;
    int even_index_limit = BUCKET_SIZE;

    for (int i = 0; i < blockDim.x; i++) {
        item_index = 2 * thread_id + 1;
        if (item_index < odd_index_limit) { // is it correct ??
            if (shared_bucket[item_index] > shared_bucket[item_index + 1]) {
                SWAP_FLOATS(shared_bucket[item_index], shared_bucket[item_index + 1]);
            }
        }
        __syncthreads();
        item_index = 2 * thread_id;
        if (thread_id < even_index_limit) { // is it correct ??
            if (shared_bucket[item_index] > shared_bucket[item_index + 1]) {
                SWAP_FLOATS(shared_bucket[item_index], shared_bucket[item_index + 1]);
            }
        }
        __syncthreads();
    }

    // write result back

    item_index = 2 * thread_id + begin_position_for_bucket[bucket_index];
    if (item_index - begin_position_for_bucket[bucket_index] < bucket_size) {
        buckets[item_index] = shared_bucket[2 * thread_id];

    }

    item_index = 2 * thread_id + 1 + begin_position_for_bucket[bucket_index];
    if (item_index - begin_position_for_bucket[bucket_index] < bucket_size) {
        buckets[item_index] = shared_bucket[2 * thread_id + 1];
    }

    __syncthreads(); // why?
}




//
// gpu_bucket_sort description:
//  data_device -- initial array allocated for GPU usage;
//            n -- amount of items in data array;
//
__host__ void gpu_bucket_sort(float *data_device, int n) {

#ifdef DEBUG
    depth_inc();
    print_depth_space();
    printf("BEGIN SORT\n");
#endif

    // find min data value
    float min = FLT_MAX;
    recursive_gpu_reduce_min(data_device, n, &min);
    CSC(cudaGetLastError());

    // find max data value
    float max = -FLT_MAX;
    recursive_gpu_reduce_max(data_device, n, &max);
    CSC(cudaGetLastError());


#ifdef DEBUG
    print_depth_space();
    printf("MIN = %f, MAX = %f\n", min, max);
#endif

    // check for already sorted array
    if (fabs(min - max) < EPS) {
#ifdef DEBUG
        depth_dec();
#endif
        return;
    }

    // make splits
    int splits_count = n / SPLIT_SIZE + 1;

#ifdef DEBUG
    print_depth_space();
    printf("splits_count = %d\n", splits_count);
#endif

    // create size_of_split on gpu
    int *size_of_split_device = NULL;
    CSC(cudaMalloc((void **)&size_of_split_device, splits_count * sizeof(int)));
    CSC(cudaMemset(size_of_split_device, 0, splits_count * sizeof(int)));
    CSC(cudaGetLastError());

    // calculate splits sizes with histogram
    gpuHistogramCalculateSplitsSizes <<<GRID_SIZE, BLOCK_SIZE>>> (data_device, n, size_of_split_device, min, max, splits_count);
    CSC(cudaGetLastError());


#ifdef DEBUG // check size_of_split array

    // int *size_of_split = (int *)malloc(splits_count * sizeof(int));
    // memset(size_of_split, 0, splits_count * sizeof(int));
    //
    // CSC(cudaMemcpy(size_of_split, size_of_split_device, splits_count * sizeof(int), cudaMemcpyDeviceToHost));
    // CSC(cudaGetLastError());
    //
    // print_depth_space();
    // printf("size_of_split: ");
    // print_array(size_of_split, splits_count);
    //
    // free(size_of_split);

#endif

    // calculate splits begin position with scan
    int *begin_position_for_split_device = NULL;
    CSC(cudaMalloc((void **)&begin_position_for_split_device, splits_count * sizeof(int)));
    CSC(cudaGetLastError());

    recursive_gpu_scan(size_of_split_device, n, begin_position_for_split_device);
    CSC(cudaGetLastError());


#ifdef DEBUG // check begin_position_for_split

    // int *begin_position_for_split = (int *)malloc(splits_count * sizeof(int));
    // CSC(cudaMemcpy(begin_position_for_split, begin_position_for_split_device, splits_count * sizeof(int), cudaMemcpyDeviceToHost));
    // CSC(cudaGetLastError());
    //
    // print_depth_space();
    // printf("begin_position_for_split: ");
    // print_array(begin_position_for_split, splits_count);
    //
    // free(begin_position_for_split);

#endif



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


#ifdef DEBUG // check splits array

    // float *splits = (float *)malloc(n * sizeof(float));
    // CSC(cudaMemcpy(splits, splits_device, n * sizeof(float), cudaMemcpyDeviceToHost));
    // CSC(cudaGetLastError());
    //
    // print_depth_space();
    // printf("splits: ");
    // print_array(splits, n);
    //
    // free(splits);

#endif




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

#ifdef DEBUG
        // print_depth_space();
        // printf("split_size = %d\n", split_size);
        // print_depth_space();
        // printf("split_begin_position = %d\n", split_begin_position);
#endif

        if (split_size > BUCKET_SIZE) {

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

    int *begin_position_for_bucket_device = NULL;
    CSC(cudaMalloc((void **)&begin_position_for_bucket_device, buckets_count * sizeof(int)));
    CSC(cudaGetLastError());
    CSC(cudaMemcpy(begin_position_for_bucket_device, begin_position_for_bucket, buckets_count * sizeof(int), cudaMemcpyHostToDevice));
    CSC(cudaGetLastError());

    int *size_of_bucket_device = NULL;
    CSC(cudaMalloc((void **)&size_of_bucket_device, buckets_count * sizeof(int)));
    CSC(cudaGetLastError());
    CSC(cudaMemcpy(size_of_bucket_device, size_of_bucket, buckets_count * sizeof(int), cudaMemcpyHostToDevice));
    CSC(cudaGetLastError());

    // oddEvenSort <<<1024, 1024>>> (splits_device, n, buckets_count, begin_position_for_bucket_device, size_of_bucket_device);

    dim3 sortBlocks(buckets_count, 1, 1);
    dim3 sortThreads(BUCKET_SIZE, 1, 1);

    gpuOddEvenSort <<<sortBlocks, sortThreads>>> (splits_device, n, begin_position_for_bucket_device, size_of_bucket_device);
    CSC(cudaGetLastError());

    // for (int i = 0; i < buckets_count; i++) {
    //     int bucket_size = size_of_bucket[i];
    //     if (bucket_size == -1) { // already sorted
    //         continue;
    //     }
    //     float *bucket = (float *)malloc(bucket_size * sizeof(float));
    //     int bucket_begin_position = begin_position_for_bucket[i];
    //
    //     CSC(cudaMemcpy(bucket, &(splits_device[bucket_begin_position]), bucket_size * sizeof(float), cudaMemcpyDeviceToHost));
    //     CSC(cudaGetLastError());
    //
    //     odd_even_sort(bucket, bucket_size);
    //
    //     CSC(cudaMemcpy(&(splits_device[bucket_begin_position]), bucket, bucket_size * sizeof(float), cudaMemcpyHostToDevice));
    //     CSC(cudaGetLastError());
    //
    //     free(bucket);
    // }

    CSC(cudaMemcpy(data_device, splits_device, n * sizeof(float), cudaMemcpyDeviceToDevice));

    CSC(cudaFree(size_of_split_device));
    CSC(cudaFree(splits_device));
    CSC(cudaFree(begin_position_for_split_device));
    CSC(cudaFree(current_size_of_split_device));
    CSC(cudaFree(begin_position_for_bucket_device));
    CSC(cudaFree(size_of_bucket_device));
    CSC(cudaGetLastError());

    free(size_of_bucket);
    free(begin_position_for_bucket);

#ifdef DEBUG
    print_depth_space();
    printf("END SORT\n");
    depth_dec();
#endif

}


//
// bucket_sort description:
//  data -- initial array allocated for CPU usage;
//     n -- amount of items in data array;
//
__host__ void bucket_sort(float *data, int n) {
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

    int n = 0;
    float *data = read_data_as_plain_text(&n);
    // float *data = read_data(&n);

    if (n == 0) {
        free(data);
        return 0;
    }

    bucket_sort(data, n);

    // print_array(data, n);

    if (sorted(data, n)) {
        printf("--\nStatus: OK\n");
    } else {
        printf("--\nStatus: WA\n");
    }

    free(data);

    return 0;
}
