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
#define BUCKET_SIZE 1024
#define SPLIT_SIZE  512




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
    for (int i = 0; i < *n; i++) {
        scanf("%f", &data[i]);
    }
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




// =============================================================================
//                                    SCAN
// =============================================================================

// plain cpu scan
// floating numbers scan
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

/****

test
10
3 -5 6 7 4 1 8 10 2 5

****/

    int n = 0;
    float *data = read_data_as_plain_text(&n);
    // float *data = read_data(&n);

    // sorting
    bucket_sort(data, n);

    // write_data_with_size(data, n);

    // printf("\n-------------------------------\n");
    if (sorted(data, n)) {
        printf("Status: OK\n");
        printf("Max recursion depth: %d\n", max_depth);
    } else {
        printf("Status: WA\n");
    }

    return 0;
}
