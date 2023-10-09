#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAX_ITER 1000
#define WIDTH 1200
#define HEIGHT 1200

void savePGMImage(const char* filename, unsigned char* data, int width, int height) {
    FILE* outfile = fopen(filename, "wb");
    if (!outfile) {
        perror("Error opening the output file");
        exit(EXIT_FAILURE);
    }
    
    fprintf(outfile, "P5\n%d %d\n255\n", width, height);
    fwrite(data, sizeof(unsigned char), width * height, outfile);
    fclose(outfile);
}

int main() {
    clock_t start_time = clock();
    int x, y, i, iter;
    double cx, cy, zx, zy, temp;
  
    // Allocate memory for the image array
    unsigned char* image = (unsigned char*)malloc(WIDTH * HEIGHT * sizeof(unsigned char));
    if (!image) {
        perror("Error allocating memory for the image");
        return EXIT_FAILURE;
    }

    // Loop over each pixel in the image
    for (y = 0; y < HEIGHT; y++) {
        for (x = 0; x < WIDTH; x++) {
            // Map the pixel coordinates to the complex plane
            cx = (double)x / WIDTH * 3.5 - 2.5;
            cy = (double)y / HEIGHT * 2.0 - 1.0;

            // Initialize the iteration variables
            zx = 0.0;
            zy = 0.0;
            iter = 0;

            // Iterate until the maximum number of iterations is reached or the point escapes
            while (iter < MAX_ITER && zx * zx + zy * zy < 4.0) {
                temp = zx * zx - zy * zy + cx;
                zy = 2.0 * zx * zy + cy;
                zx = temp;
                iter++;
            }

            // Map the number of iterations to a color value
            if (iter == MAX_ITER) {
                image[y * WIDTH + x] = 0;
            } else {
                image[y * WIDTH + x] = (unsigned char)(255.0 * ((double)iter / MAX_ITER) * ((double)iter / MAX_ITER));
            }
        }
    }

    // Write the image to a PGM file
    savePGMImage("mandelbrotS.pgm", image, WIDTH, HEIGHT);

    // Free the memory for the image array
    free(image);

    clock_t end_time = clock();
    double elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Elapsed time: %f seconds\n", elapsed_time);

    return EXIT_SUCCESS;
}
