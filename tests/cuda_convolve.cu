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
#define B       32
#define C       20
#define SIZES   30
#define TRIALS  3
#define OPTIONS 5

#define BLOCK_SIZE 16
#define T_COUNT 16*16

#define k_s 15

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

/* Normal conv */
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

/* Conv with shared memory */
__global__ void kernel_shared_conv(double * input, double * output, double * kernel, 
                            double bias, int stride, 
                            int in_size, int out_size, int kernel_size) {
    // Read kernel into shared memory
    // Each thread read in kernel_size number of inputs into shared memory
    // To decide whcih thread reads in kernel, just use index
    // We have about 96KB of memory = 12K doublesc

    // Each output depends on kernel_size * kernel_size inputs
    // A row of output depends on kernel_size * in_size inputs

    extern __shared__ double s[];
    double * kernel_s = s;
    double * input_s = &kernel_s[kernel_size * kernel_size];

    int c = threadIdx.x;
    int r = blockIdx.x; 
    int i, j, ki = 0, kj = 0;
    double sum = 0;

    int tot_kernel_size = kernel_size * kernel_size;
    // Simple case, each thread read one kernel element
    if (tot_kernel_size <= in_size) {
        if (c < tot_kernel_size) kernel_s[c] = kernel[c];
    } else {    // Stupid case, 
        float lim = ceil((float) tot_kernel_size / in_size);
        for (i = 0; i < lim; i++) {
            int k_i = c + (i * in_size);
            if (k_i < tot_kernel_size) kernel_s[k_i] = kernel[k_i];
        }
    }
    __syncthreads();

    // Read kernel_size inputs
    for (i = 0; i < kernel_size; i++) input_s[i * in_size + c] = input[(i+r) * in_size + c];
    __syncthreads();

    // Each thread less within boudns do convo
    if (c < out_size) {
        for (i = 0; i < kernel_size; i++) {
            kj = 0;
            for (j = c; j < c + kernel_size; j++) {
                sum += kernel_s[ki * kernel_size + kj] * input_s[i * in_size + j];
                kj++;
            }
            ki++;
        }

        output[r * out_size + c] = sum + bias;
    }

};

/* Conv  with blocked shared memory */
__global__ void kernel_blocked_conv(double * input, double * output, double * kernel, 
                            double bias, int stride, 
                            int in_size, int out_size, int kernel_size) {
    /*
        We have threads equal to the the block size squared. Could be 16x16 or 32x32
        Works the same way as shared conv, but instead of reading in rows, we read in blocks,
        Hopefully this reduces redundant mem reads and increase reuse
    */
    extern __shared__ double s[];
    double * kernel_s = s;
    double * input_s = &kernel_s[kernel_size * kernel_size];

    double sum = 0;
    int i, j, ki = 0, kj = 0;
    int t_x = threadIdx.x, t_y = threadIdx.y;
    int b_x = blockIdx.x, b_y = blockIdx.y;
    int out_x = b_x * BLOCK_SIZE + t_x, out_y = b_y * BLOCK_SIZE + t_y;
    int tid = t_x + t_y * BLOCK_SIZE;   // Linear thread id
    int shared_i_size = BLOCK_SIZE + kernel_size - 1;   // The length of shared input block

    // First read in kernel
    int tot_kernel_size = kernel_size * kernel_size;
    // Simple case, some threads read one kernel element
    if (tot_kernel_size <= T_COUNT) {
        if (tid < tot_kernel_size) kernel_s[tid] = kernel[tid];
    } else {    // Stupid case, 
        float lim = ceil((float) tot_kernel_size / T_COUNT);
        for (i = 0; i < lim; i++) {
            int k_i = tid + (i * T_COUNT);
            if (k_i < tot_kernel_size) kernel_s[k_i] = kernel[k_i];
        }
    }
    __syncthreads();

    // Next read in input matrix
    /*
    Fun math: We know for a fact, that kernels are always smaller than input images,
    Therefore, if each thread read 4 elements from input, we will always read more or equal to the nubmer of actual inputs
    This allows a memory coalesceced way of reading the input matrix to shraed mem
    */
    int in_count = shared_i_size * shared_i_size;
    int b_x_i = b_x * BLOCK_SIZE;
    int b_y_i = b_y * BLOCK_SIZE * in_size;
    for (i = 0; i < 4; i++) {
        int in_i = i * T_COUNT + tid;
        if (in_i < in_count) input_s[in_i] = input[(in_i % shared_i_size) + ((in_i / shared_i_size) * in_size) 
        + (b_x_i) + (b_y_i)];
    }
    __syncthreads();
    
    // Now each thread does conv for one element
    for (i = t_y; i < t_y + kernel_size; i++) {
        kj = 0;
        for (j = t_x; j < t_x + kernel_size; j++) {
            sum += kernel_s[ki * kernel_size + kj] * input_s[i * shared_i_size + j];
            kj++;
        }
        ki++;
    }

    output[out_y * out_size + out_x] = sum + bias;                    

}

/* Conv with shared memory and unroll */
/* Conv with shared memory */
__global__ void kernel_shared_unroll_conv(double * input, double * output, double * kernel, 
                            double bias, int stride, 
                            int in_size, int out_size, int kernel_size) {
    // Read kernel into shared memory
    // Each thread read in kernel_size number of inputs into shared memory
    // To decide whcih thread reads in kernel, just use index
    // We have about 96KB of memory = 12K doublesc

    // Each output depends on kernel_size * kernel_size inputs
    // A row of output depends on kernel_size * in_size inputs

    extern __shared__ double s[];
    double * kernel_s = s;
    double * input_s = &kernel_s[kernel_size * kernel_size];

    int c = threadIdx.x;
    int r = blockIdx.x; 
    int i;

    int tot_kernel_size = kernel_size * kernel_size;
    // Simple case, each thread read one kernel element
    if (tot_kernel_size <= in_size) {
        if (c < tot_kernel_size) kernel_s[c] = kernel[c];
    } else {    // Stupid case, 
        float lim = ceil((float) tot_kernel_size / in_size);
        for (i = 0; i < lim; i++) {
            int k_i = c + (i * in_size);
            if (k_i < tot_kernel_size) kernel_s[k_i] = kernel[k_i];
        }
    }
    __syncthreads();

    // Read kernel_size inputs
    for (i = 0; i < kernel_size; i++) input_s[i * in_size + c] = input[(i+r) * in_size + c];
    __syncthreads();


    double acc1 = 0;
    double acc2 = 0;
    double acc3 = 0;
    double acc4 = 0;

    // Each thread less within boudns do convo
    if (c < out_size) {

        for (i = 0; i < kernel_size * kernel_size - 3; i+=4) {
            acc1 += kernel_s[i] * input_s[c + ((i / kernel_size) * in_size) + (i % kernel_size)];
            acc2 += kernel_s[i + 1] * input_s[c + (((i + 1) / kernel_size) * in_size) + ((i + 1) % kernel_size)];
            acc3 += kernel_s[i + 2] * input_s[c + (((i + 2) / kernel_size) * in_size) + ((i + 2) % kernel_size)];
            acc4 += kernel_s[i + 3] * input_s[c + (((i + 3) / kernel_size) * in_size) + ((i + 3) % kernel_size)];
        }

        // Remainders
        for (;i < kernel_size * kernel_size; i++) {
            acc1 += kernel_s[i] * input_s[c + ((i / kernel_size) * in_size) + (i % kernel_size)];
        }

        output[r * out_size + c] = acc1 + acc2 + acc3 + acc4 + bias;
    }

};

/* --- End CUDA functions --- */

int main(int argc, char **argv) {
    // Init vars
    double * in_h, * out_h, * ker_h;
    double * in_d, * ker_d;
    double * out_d;
    double * out_CPU;	// CPU output matrix
    double bias = rand_double(R_MIN, R_MAX);
    struct timespec time_start, time_stop;
    double measurements[OPTIONS * SIZES]; // To contain time: GPU, Shared GPU CPU
    int mat_sizes[SIZES];

    // Clear time
    for (int i = 0; i < OPTIONS * SIZES; i++) measurements[i] = 0;

    /*
    First idea:
        Assign each block a row
        Get as many blocks as there are rows

    Second idea: Apply shared memory

    Third idea: shared memory but in blocks, then CPU conv over the remainders

    Fourth idea: Better memory coalescense
    */
    for (int i = 0; i < SIZES; i++) {
        int in_s, out_s;
        in_s = A * i * i + B * i + C;
        out_s = in_s - k_s + 1;
        mat_sizes[i] = in_s;

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

        for (int t = 0; t < TRIALS; t++) {
            // Init matrix
            init_mat(in_h, in_s);
            init_mat(ker_h, k_s);

            // Run CPU code
            clock_gettime(CLOCK_REALTIME, &time_start);
            convolve(in_h, out_CPU, ker_h, bias, STRIDE, in_s, out_s, k_s);
            clock_gettime(CLOCK_REALTIME, &time_stop);
            measurements[i * OPTIONS] += interval(time_start, time_stop);

            // --- Normal cov kernel ---
            {
                clock_gettime(CLOCK_REALTIME, &time_start);

                // Copy data to GPU
                cudaMemcpy(in_d, in_h, in_alloc_size, cudaMemcpyHostToDevice);
                cudaMemcpy(ker_d, ker_h, ker_alloc_size, cudaMemcpyHostToDevice);


                // Shared cov kernel
                dim3 griddim(out_s);
                dim3 blockdim(out_s);
                kernel_conv<<<griddim, blockdim>>>(in_d, out_d, ker_d, bias, STRIDE, in_s, out_s, k_s);
                // Copy back
                cudaMemcpy(out_h, out_d, out_alloc_size, cudaMemcpyDeviceToHost);
                // printf("Done with Conv on GPU\n");

                clock_gettime(CLOCK_REALTIME, &time_stop);
                measurements[i * OPTIONS + 1] += interval(time_start, time_stop);

                // Compare accuracy
                compare_mat(out_CPU, out_h, out_s, out_s);
            }

            // --- Shared cov kernel ---
            {
                clock_gettime(CLOCK_REALTIME, &time_start);

                // Copy data to GPU
                cudaMemcpy(in_d, in_h, in_alloc_size, cudaMemcpyHostToDevice);
                cudaMemcpy(ker_d, ker_h, ker_alloc_size, cudaMemcpyHostToDevice);


                // Shared cov kernel
                dim3 griddim(out_s);
                dim3 blockdim(in_s);
                kernel_shared_conv<<<griddim, blockdim, (k_s * k_s + k_s * in_s) * sizeof(double)>>>(in_d, out_d, ker_d, bias, STRIDE, in_s, out_s, k_s);
                // Copy back
                cudaMemcpy(out_h, out_d, out_alloc_size, cudaMemcpyDeviceToHost);
                // printf("Done with Conv on GPU\n");

                clock_gettime(CLOCK_REALTIME, &time_stop);
                measurements[i * OPTIONS + 2] += interval(time_start, time_stop);

                // Compare accuracy
                compare_mat(out_CPU, out_h, out_s, out_s);
            }

            // --- Shared with unroll ---
            {
                clock_gettime(CLOCK_REALTIME, &time_start);

                // Copy data to GPU
                cudaMemcpy(in_d, in_h, in_alloc_size, cudaMemcpyHostToDevice);
                cudaMemcpy(ker_d, ker_h, ker_alloc_size, cudaMemcpyHostToDevice);


                // Shared cov kernel
                dim3 griddim(out_s);
                dim3 blockdim(in_s);
                kernel_shared_unroll_conv<<<griddim, blockdim, (k_s * k_s + k_s * in_s) * sizeof(double)>>>(in_d, out_d, ker_d, bias, STRIDE, in_s, out_s, k_s);
                // Copy back
                cudaMemcpy(out_h, out_d, out_alloc_size, cudaMemcpyDeviceToHost);
                // printf("Done with Conv on GPU\n");

                clock_gettime(CLOCK_REALTIME, &time_stop);
                measurements[i * OPTIONS + 3] += interval(time_start, time_stop);

                // Compare accuracy
                compare_mat(out_CPU, out_h, out_s, out_s);
            }

            // --- Blocked cov kernel ---
            {
                clock_gettime(CLOCK_REALTIME, &time_start);

                // Copy data to GPU
                cudaMemcpy(in_d, in_h, in_alloc_size, cudaMemcpyHostToDevice);
                cudaMemcpy(ker_d, ker_h, ker_alloc_size, cudaMemcpyHostToDevice);


                // Blocked cov kernel
                dim3 griddim(out_s / BLOCK_SIZE, out_s / BLOCK_SIZE);
                dim3 blockdim(BLOCK_SIZE, BLOCK_SIZE);
                kernel_blocked_conv<<<griddim, blockdim, (k_s * k_s + (k_s - 1 + BLOCK_SIZE) * (k_s - 1 + BLOCK_SIZE)) * sizeof(double)>>>(in_d, out_d, ker_d, bias, STRIDE, in_s, out_s, k_s);
                // Copy back
                cudaMemcpy(out_h, out_d, out_alloc_size, cudaMemcpyDeviceToHost);
                // printf("Done with Conv on GPU\n");

                clock_gettime(CLOCK_REALTIME, &time_stop);
                measurements[i * OPTIONS + 4] += interval(time_start, time_stop);

                // Compare accuracy
                compare_mat(out_CPU, out_h, out_s, out_s);
            }
        }

        // Deallocate GPU mem
        cudaFree(in_d);
        cudaFree(out_d);
        cudaFree(ker_d);
    }

    // Average times
    for (int i = 0; i < OPTIONS * SIZES; i++) measurements[i] /= TRIALS;

    // Compare time
    printf("Tested for kernel size %d, stride %d\n", k_s, STRIDE);
    printf("Time measurements (s): Matrix Size, CPU, GPU, Shared GPU, Shared GPU + 2x Unroll, Blocked GPU\n");
    for (int i = 0; i < SIZES; i++) printf("%d, %f, %f, %f, %f, %f\n", 
        mat_sizes[i], measurements[i*OPTIONS], 
        measurements[i*OPTIONS + 1], 
        measurements[i*OPTIONS + 2],
        measurements[i*OPTIONS + 3],
        measurements[i*OPTIONS + 4]);
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

    // if (total_err < 0.000001) printf("No error!\n");
    
    if (total_err > 0.000001) {
        printf("Errors founded for size %d!\n", r_len);
        printf("Average absolute err per element is: %f\n", avg_err);
        printf("Largest absolute err between 2 elements was: %f\n", largest_err);
        printf("Total err is: %.5f\n", total_err);
    }
}