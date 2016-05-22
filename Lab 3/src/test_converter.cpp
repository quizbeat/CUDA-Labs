//
//  test_converter.cpp
//  CUDA-Lab-3
//
//  Created by Nikita Makarov on 07/05/16.
//  Copyright © 2016 Nikita Makarov. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <cstring>

int main(int argc, char const *argv[]) {

    // program input format:
    // binary -> text: ./test_converter.out -text bin.data text.txt
    // text -> binary: ./test_converter.out -bin  text.txt bin.data

    if (argc != 4) {
        std::cout << "Wrong arguments count." << std::endl;
        return 0;
    }

    if (strcmp(argv[1], "-text") == 0) {
        // convert binary data to text data
        FILE *binary_file = fopen(argv[2], "rb");
        // read numbers count (int value)
        int n = 0;
        fread(&n, sizeof(int), 1, binary_file);
        // read numbers
        float *numbers = (float *)malloc(n * sizeof(float));
        fread(numbers, sizeof(float), n, binary_file);
        // write numbers to text file
        std::ofstream text_file(argv[3]);
        text_file << n << '\n';
        for (int i = 0; i < n; i++) {
            text_file << numbers[i] << " ";
        }
        text_file << '\n';
        // close files
        text_file.close();
        fclose(binary_file);

    } else if (strcmp(argv[1], "-bin") == 0) {
        // convert text data to binary data
        std::ifstream text_file(argv[2]);
        // read numbers count
        int n = 0;
        text_file >> n;
        // read numbers
        float *numbers = (float *)malloc(n * sizeof(float));
        for (int i = 0; i < n; i++) {
            text_file >> numbers[i];
        }
        // write numbers count to binary file
        FILE *binary_file = fopen(argv[3], "wb");
        fwrite(&n, sizeof(int), 1, binary_file);
        // write numbers to binary file
        fwrite(numbers, sizeof(float), n, binary_file);
        // close files
        fclose(binary_file);
        text_file.close();

    } else {
        std::cout << "Invalid option." << std::endl;
        return 0;
    }

    return 0;
}
