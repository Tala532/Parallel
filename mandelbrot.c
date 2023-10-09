#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define WIDTH 800
#define HEIGHT 800
#define MAX_ITER 1000
#define X_MIN -2.0
#define X_MAX 1.0
#define Y_MIN -1.5
#define Y_MAX 1.5

int mandelbrot(double real, double imag) {
    int n;
    double r = 0.0;
    double i = 0.0;

    for (n = 0; n < MAX_ITER; n++) {
        double r2 = r * r;
        double i2 = i * i;
        if (r2 + i2 > 4.0)
            return n;
        i = 2.0 * r * i + imag;
        r = r2 - i2 + real;
    }

    return MAX_ITER;
}

void savePGM(const char *filename, int *data, int width, int height) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        perror("Error opening file");
        return;
    }

    fprintf(file, "P2\n");
    fprintf(file, "%d %d\n", width, height);
    fprintf(file, "255\n");

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            fprintf(file, "%d ", data[i * width + j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}



int main(int argc, char *argv[]) {
    int rank, size;
    double start_time, end_time;
     double computation_time, communication_time;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    start_time = MPI_Wtime();
    int rows_per_process = HEIGHT / size;
    int start_row = rank * rows_per_process;
    int end_row = (rank + 1) * rows_per_process;

    if (rank == size - 1) {
        end_row = HEIGHT;
    }

    int local_result[WIDTH * (end_row - start_row)];
 double computation_start_time = MPI_Wtime();
    for (int y = start_row; y < end_row; y++) {
        for (int x = 0; x < WIDTH; x++) {
            double real = (x * (X_MAX - X_MIN) / (WIDTH - 1)) + X_MIN;
            double imag = (y * (Y_MAX - Y_MIN) / (HEIGHT - 1)) + Y_MIN;
            int value = mandelbrot(real, imag);
            local_result[(y - start_row) * WIDTH + x] = value;
        }
    }
     double computation_end_time = MPI_Wtime();
      computation_time = computation_end_time - computation_start_time;

    if (rank == 0) {
        int *global_result = (int *)malloc(sizeof(int) * WIDTH * HEIGHT);
        MPI_Gather(local_result, WIDTH * (end_row - start_row), MPI_INT, global_result, WIDTH * (end_row - start_row), MPI_INT, 0, MPI_COMM_WORLD);

        // Save the global_result as a PGM image
        savePGM("mandelbrot.pgm", global_result, WIDTH, HEIGHT);

        free(global_result);
    } else {
        MPI_Gather(local_result, WIDTH * (end_row - start_row), MPI_INT, NULL, 0, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    end_time = MPI_Wtime(); // Record the end time

   if (rank == 0) {
        printf("Total execution time: %lf seconds\n", end_time - start_time);
        printf("Computation time: %lf seconds\n", computation_time);
        printf("Communication time: %lf seconds\n", end_time - start_time - computation_time);
    }
    

    return 0;
}
