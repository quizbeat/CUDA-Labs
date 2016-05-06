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

#define MIN(a, b) (a < b ? a : b)
#define MAX(a, b) (a > b ? a : b)
#define IDX(i, j, w, h) (MAX(MIN(i, h - 1), 0) * w + MAX(MIN(j, w - 1), 0))

typedef struct {
    unsigned char r, g, b, a;
} Pixel;

typedef struct {
    int w;
    int h;
    Pixel *pixels;
} Image;

Image *read_image_at_path(const char *path) {
    FILE *image_file = fopen(path, "rb");
    
    if (!image_file) {
        printf("Error: can't open file.");
        return NULL;
    }
    
    Image *image = (Image *)malloc(sizeof(Image));
    
    // read width and height from file
    fread(&(image->w), sizeof(image->w), 1, image_file);
    fread(&(image->h), sizeof(image->h), 1, image_file);
    
    // alloc memory for pixels
    image->pixels = (Pixel *)malloc(image->w * image->h * sizeof(Pixel));
    
    // read pixels
    for (int i = 0; i < image->h; i++) {
        for (int j = 0; j < image->w; j++) {
            int pixel_index = IDX(i, j, image->w, image->h);
            fread(&(image->pixels[pixel_index].r), sizeof(image->pixels[pixel_index].r), 1, image_file);
            fread(&(image->pixels[pixel_index].g), sizeof(image->pixels[pixel_index].g), 1, image_file);
            fread(&(image->pixels[pixel_index].b), sizeof(image->pixels[pixel_index].b), 1, image_file);
            fread(&(image->pixels[pixel_index].a), sizeof(image->pixels[pixel_index].a), 1, image_file);
        }
    }
    
    fclose(image_file);
    
    return image;
}

void write_image_to_file(Image *image, const char *path) {
    FILE *image_file = fopen(path, "wb");
    
    // write dimensions
    fwrite(&(image->w), sizeof(image->w), 1, image_file);
    fwrite(&(image->h), sizeof(image->h), 1, image_file);
    
    // write pixels
    for (int i = 0; i < image->h; i++) {
        for (int j = 0; j < image->w; j++) {
            int pixel_index = IDX(i, j, image->w, image->h);
            fwrite(&(image->pixels[pixel_index].r), sizeof(image->pixels[pixel_index].r), 1, image_file);
            fwrite(&(image->pixels[pixel_index].g), sizeof(image->pixels[pixel_index].g), 1, image_file);
            fwrite(&(image->pixels[pixel_index].b), sizeof(image->pixels[pixel_index].b), 1, image_file);
            fwrite(&(image->pixels[pixel_index].a), sizeof(image->pixels[pixel_index].a), 1, image_file);
        }
    }
    
    fclose(image_file);
}

double grayscale_value_for_pixel(Pixel pixel) {
    return (0.299 * pixel.r) + (0.587 * pixel.g) + (0.114 * pixel.b);
}

unsigned char Sobel_pixel(Image *image, int i, int j) {
    
    double G_x_component_1 = 0.0;
    double G_x_compinent_2 = 0.0;
    
    for (int p = -1; p < 2; p++) {
        
        int pixel_index_top = IDX(i - 1, j + p, image->w, image->h);
        int pixel_index_bottom = IDX(i + 1, j + p, image->w, image->h);
        
        double gray_top = grayscale_value_for_pixel(image->pixels[pixel_index_top]);
        double gray_bottom = grayscale_value_for_pixel(image->pixels[pixel_index_bottom]);
        
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
        
        int pixel_index_left = IDX(i + p, j - 1, image->w, image->h);
        int pixel_index_right = IDX(i + p, j + 1, image->w, image->h);
        
        double gray_left = grayscale_value_for_pixel(image->pixels[pixel_index_left]);
        double gray_right = grayscale_value_for_pixel(image->pixels[pixel_index_right]);
        
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
    
    return normalized_sobel;
}

int main(int argc, const char * argv[]) {
    
    std::string input_path;
    std::string output_path;
    
    std::cin >> input_path;
    std::cin >> output_path;
    
    Image *image = read_image_at_path(input_path.c_str());
            
    Image *filtered = (Image *)malloc(sizeof(Image));
    filtered->w = image->w;
    filtered->h = image->h;
    filtered->pixels = (Pixel *)malloc(filtered->w * filtered->h * sizeof(Pixel));
    
    for (int i = 0; i < filtered->h; i++) {
        for (int j = 0; j < filtered->w; j++) {
            unsigned char pixel_value = Sobel_pixel(image, i, j);
            filtered->pixels[IDX(i, j, filtered->w, filtered->h)].r = pixel_value;
            filtered->pixels[IDX(i, j, filtered->w, filtered->h)].g = pixel_value;
            filtered->pixels[IDX(i, j, filtered->w, filtered->h)].b = pixel_value;
            filtered->pixels[IDX(i, j, filtered->w, filtered->h)].a = 0;
        }
    }
    
    write_image_to_file(filtered, output_path.c_str());
    
    free(filtered->pixels);
    free(filtered);
    
    free(image->pixels);
    free(image);
    
    return 0;
}
