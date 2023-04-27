#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <time.h>

void convolve(double * input, double * output, double * kernel, double bias, int stride, int in_size, int out_size, int kernel_size);


/* --- Interval code --- */
double interval(struct timespec start, struct timespec end)
{
  struct timespec temp;
  temp.tv_sec = end.tv_sec - start.tv_sec;
  temp.tv_nsec = end.tv_nsec - start.tv_nsec;
  if (temp.tv_nsec < 0) {
    temp.tv_sec = temp.tv_sec - 1;
    temp.tv_nsec = temp.tv_nsec + 1000000000;
  }
  return (((double)temp.tv_sec) + ((double)temp.tv_nsec)*1.0e-9);
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