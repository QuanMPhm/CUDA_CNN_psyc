/*
    nvcc cuda_convolve.cu -o ./bin/cuda_convolve
*/

#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <time.h>

#define R_MIN 1.0
#define R_MAX -1.0

#define STRIDE 1

#define A       0
#define B       16
#define C       28
#define OPTIONS 20


#define k_s 5

void convolve(double * input, double * output, double * kernel, double bias, int stride, int in_size, int out_size, int kernel_size);
void init_mat(double * input, int len);    // Generate 2-D matrix, give allocated pointer
double rand_double(double min, double max);     // Generate random float in interval
void compare_mat(double * a, double * b, int r_len, int c_len); // Matrix comparison

/* --- Interval code --- */
double interval(struct timespec start, struct timespec end) {
  struct timespec temp;
  temp.tv_sec = end.tv_sec - start.tv_sec;
  temp.tv_nsec = end.tv_nsec - start.tv_nsec;
  if (temp.tv_nsec < 0) {
    temp.tv_sec = temp.tv_sec - 1;
    temp.tv_nsec = temp.tv_nsec + 1000000000;
  }
  return (((double)temp.tv_sec) + ((double)temp.tv_nsec)*1.0e-9);
}

/* --- CUDA functions --- */

__global__ void kernel_conv(double * input, double * output, double * kernel, 
                            double bias, int stride, 
                            int in_size, int out_size, int kernel_size) {
    // First idea is to...
    // Each block is responsible for one row. Considering max thread/block is 1024, we'll limit to 1024 for now
    // Each thread responsible for one element of output matrix
    int c = threadIdx.x;
    int r = blockIdx.x; 
    int i, j, ki = 0, kj = 0;
    double sum = 0;

    for (i = r; i < r + kernel_size; i++) {
        kj = 0;
        for (j = c; j < c + kernel_size; j++) {
            sum += kernel[ki * kernel_size + kj] * input[i * in_size + j];
            kj++;
        }
        ki++;
    }

    output[r * out_size + c] = sum + bias;
};



/* --- End CUDA functions --- */

int main(int argc, char **argv) {
    // Init vars
    double * in_h, * out_h, * ker_h;
    double * in_d, * out_d, * ker_d;
    double * out_CPU;	// CPU output matrix
    double bias = rand_double(R_MIN, R_MAX);
    struct timespec time_start, time_stop;
    double measurements[2 * OPTIONS]; // To contain time: GPU, CPU
    int mat_sizes[OPTIONS];

    // struct timespec time_start_2, time_stop_2;

    /*
    Get grid and block dimensions
    First idea:
        Assign each block a row
        Get as many blocks as there are rows

    Second idea: Apply local memory

    Third idea: Local memory + unrolling

    Fourth idea: Better memory coalescense
    */
    for (int i = 0; i < OPTIONS; i++) {
        int in_s, out_s;
        in_s = A * i * i + B * i + C;
        out_s = in_s - k_s + 1;
        mat_sizes[i] = in_s;
        int block_n = out_s;
        int thread_n = out_s;
        dim3 griddim(block_n);
        dim3 blockdim(thread_n);

        // Allocate arrays
        int in_alloc_size = sizeof(double) * in_s * in_s;
        int out_alloc_size = sizeof(double) * out_s * out_s;
        int ker_alloc_size = sizeof(double) * k_s * k_s;

        in_h = (double *) malloc(in_alloc_size);
        out_h = (double *) malloc(out_alloc_size);
        ker_h = (double *) malloc(ker_alloc_size);
        out_CPU = (double *) malloc(out_alloc_size);

        cudaMalloc((void **) &in_d, in_alloc_size);
        cudaMalloc((void **) &out_d, out_alloc_size);
        cudaMalloc((void **) &ker_d, ker_alloc_size);

        // Init matrix
        init_mat(in_h, in_s);
        init_mat(ker_h, k_s);

        clock_gettime(CLOCK_REALTIME, &time_start);

        // Copy data to GPU
        cudaMemcpy(in_d, in_h, in_alloc_size, cudaMemcpyHostToDevice);
        cudaMemcpy(ker_d, ker_h, ker_alloc_size, cudaMemcpyHostToDevice);

        // Run kernel
        kernel_conv<<<griddim, blockdim>>>(in_d, out_d, ker_d, bias, STRIDE, in_s, out_s, k_s);
        // Copy back
        cudaMemcpy(out_h, out_d, out_alloc_size, cudaMemcpyDeviceToHost);
        // printf("Done with Conv on GPU\n");

        clock_gettime(CLOCK_REALTIME, &time_stop);
        measurements[i * 2] = interval(time_start, time_stop);

        // Deallocate GPU mem
        cudaFree(in_d);
        cudaFree(out_d);
        cudaFree(ker_d);

        // Run CPU code
        clock_gettime(CLOCK_REALTIME, &time_start);
        convolve(in_h, out_CPU, ker_h, bias, STRIDE, in_s, out_s, k_s);
        clock_gettime(CLOCK_REALTIME, &time_stop);
        measurements[i * 2 + 1] = interval(time_start, time_stop);

        // Compare accuracy
        compare_mat(out_CPU, out_h, out_s, out_s);
    }

    // Compare time
    printf("Time measurements (s): Matrix Size, GPU (w/ transfers, CPU\n");
    for (int i = 0; i < OPTIONS; i++) printf("%d, %f, %f\n", mat_sizes[i], measurements[i*2], measurements[i*2 + 1]);
    printf("\n");

}

void convolve(double * input, double * output, double * kernel, double bias, int stride,  int in_size, int out_size, int kernel_size) {
    int i, j, ii, jj, ki, kj;
    double sum;
    // For each output entry:
    for (i = 0; i < out_size; i++) {
        for (j = 0; j < out_size; j++) {
            sum = 0;
            ki = 0; kj = 0;

            // Convolve
            for (ii = i; ii < i + kernel_size; ii++) {
                kj = 0;
                for (jj = j; jj < j + kernel_size; jj++) {
                    sum += kernel[ki * kernel_size + kj] * input[ii * in_size + jj];
                    kj++;
                }
                ki++;
            }

            output[i * out_size + j] = sum + bias;
        }
    }

}

double rand_double(double min, double max) {
    double random = ((double) rand()) / (double) RAND_MAX;
    double diff = max - min;
    double r = random * diff;
    return min + r;
}

void init_mat(double * input, int len) {
    for (int i = 0; i < len; i++) {
        for (int j = 0; j < len; j++) {
            input[i * len + j] = rand_double(R_MIN, R_MAX);
        }
    }
}

void compare_mat(double * a, double * b, int r_len, int c_len) {
    double avg_err;
    double largest_err = 0;
    double total_err = 0;

    for (int i = 0; i < c_len; i++) {
        for (int j = 0; j < r_len; j++) {
            double err = abs(a[i * r_len + j] - b[i * r_len + j]);
            total_err += err;
            if (err > largest_err) largest_err = err;
        }
    }

    avg_err = total_err / (r_len * c_len);

    if (total_err < 0.000001) printf("No error!\n");
    else {
        printf("Errors founded for size %d!\n", r_len);
        printf("Average absolute err per element is: %f\n", avg_err);
        printf("Largest absolute err between 2 elements was: %f\n", largest_err);
        printf("Total err is: %.5f\n", total_err);
    }
}